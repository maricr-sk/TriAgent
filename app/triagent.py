"""
TriAgent: a command-line chat that runs the v0.2 prompt and tools.
"""

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import anthropic
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config"
PROMPT_PATH = CONFIG / "system_prompt.txt"
TOOLS_PATH = CONFIG / "tools.json"
INTAKE_SCHEMA_PATH = CONFIG / "intake_schema.json"
RULES_PATH = CONFIG / "triage_rules.yaml"

MODEL = os.environ.get("TRIAGENT_MODEL", "claude-opus-5-5")
MAPS_KEY = os.environ.get("MAPS_API_KEY")
MILE_IN_METERS = 1609.34


# ---------------------------------------------------------------------------
# Loading the prompt, tools and rules
# ---------------------------------------------------------------------------

RULES = yaml.safe_load(RULES_PATH.read_text())
LEVELS = {level["id"]: level for level in RULES["care_levels"]}
RANK_TO_LEVEL = {level["rank"]: level["id"] for level in RULES["care_levels"]}
RED_FLAGS = {flag["id"]: flag for flag in RULES["red_flags"]}
COMPLAINTS = {rule["id"]: rule for rule in RULES["symptom_rules"]}


def load_tools():
    """Read tools.json and put the real intake schema in place of its $ref."""
    tools = json.loads(TOOLS_PATH.read_text())
    schema = json.loads(INTAKE_SCHEMA_PATH.read_text())
    for key in ("$schema", "$id", "title", "examples"):  # documentation only
        schema.pop(key, None)
    signs = RULES["escalation_signs"]
    field = schema["properties"]["escalation_signs"]
    field["items"]["enum"] = list(signs)  # the rules file is the source of truth
    field["description"] += " Signs: " + "; ".join(f"{k}: {v}" for k, v in signs.items())
    for tool in tools:
        if tool["name"] == "record_intake":
            tool["input_schema"] = schema
    return tools


# ---------------------------------------------------------------------------
# Tool 1: record_intake
# ---------------------------------------------------------------------------

def record_intake(intake, new_facts):
    """Merge new facts into the intake record. Lists grow, objects update,
    everything else is replaced. Returns the record and what's still missing."""
    for key, value in new_facts.items():
        if isinstance(value, list):
            existing = intake.setdefault(key, [])
            existing.extend(item for item in value if item not in existing)
        elif isinstance(value, dict) and isinstance(intake.get(key), dict):
            intake[key].update(value)
        else:
            intake[key] = value

    missing = []
    if not intake.get("symptoms") and not intake.get("red_flags"):
        missing.append("symptoms")
    for field in ("location", "travel_radius"):
        if not intake.get(field):
            missing.append(field)
    return {"intake": intake, "missing_required": missing}


# ---------------------------------------------------------------------------
# Tool 2: get_care_level (applies triage_rules.yaml)
# ---------------------------------------------------------------------------

def rank(level_id):
    return LEVELS[level_id]["rank"]


def get_care_level(intake):
    # 1. Red flags first. Any match ends triage at that flag's level.
    flags = [RED_FLAGS[f] for f in intake.get("red_flags", []) if f in RED_FLAGS]
    if intake.get("feels_like_emergency"):
        flags.append({"id": "feels_like_emergency", "level": "call_911",
                      "signs": ["person says it feels like an emergency"]})
    if flags:
        top = max(flags, key=lambda f: rank(f["level"]))
        return {
            "level": top["level"],
            "label": LEVELS[top["level"]]["label"],
            "reason": f"Matches emergency signs: {top['signs'][0]}.",
            "rule_id": top["id"],
            "watch_for": [],
            "missing": [],
        }

    # 2. The symptom rule. With several complaints, the highest level wins.
    complaint_ids = [s.get("complaint_id", "other") for s in intake.get("symptoms", [])]
    known = [COMPLAINTS[c] for c in complaint_ids if c in COMPLAINTS]
    if not known:
        # No matching rule: tie_break says pick the higher level when unsure.
        return {
            "level": "clinic_or_telehealth",
            "label": LEVELS["clinic_or_telehealth"]["label"],
            "reason": "No exact rule matched, so the safer level was chosen.",
            "rule_id": "tie_break",
            "watch_for": [],
            "missing": [],
        }
    results = [apply_complaint(rule, intake) for rule in known]
    return max(results, key=lambda r: rank(r["level"]))


def apply_complaint(rule, intake):
    """Default level -> escalate_if / deescalate_if -> modifiers."""
    level, fired = rule["default_level"], rule["id"]
    escalations = [e for e in rule.get("escalate_if", [])
                   if condition_holds(e["if"], intake, rule)]
    if escalations:
        top = max(escalations, key=lambda e: rank(e["to"]))
        if rank(top["to"]) > rank(level):
            level, fired = top["to"], top["id"]
    else:
        for line in rule.get("deescalate_if", []):
            if condition_holds(line["if"], intake, rule):
                level, fired = line["to"], line["id"]
                break

    # Modifiers: at most +1 in total, capped at er. Skip ones an escalation
    # line already accounts for.
    covered = {m for e in escalations for m in e.get("includes_modifiers", [])}
    modifiers = [m["id"] for m in RULES["modifiers"]
                 if m["id"] not in covered and condition_holds(m["if"], intake, rule)]
    if modifiers and rank(level) < rank("er"):
        level = RANK_TO_LEVEL[rank(level) + 1]

    reason = f"Based on the {rule['label'].lower()} guidance"
    if fired != rule["id"]:
        line = next(e for e in rule["escalate_if"] + rule["deescalate_if"] if e["id"] == fired)
        reason += f" ({line['when']})"
    if modifiers:
        reason += f", one step higher for: {', '.join(modifiers)}"
    return {
        "level": level,
        "label": LEVELS[level]["label"],
        "reason": reason + ".",
        "rule_id": fired,
        "modifiers": modifiers,
        "watch_for": [e["when"] for e in rule.get("escalate_if", [])
                      if rank(e["to"]) > rank(level)],
        "missing": missing_answers(rule, intake, level),
    }


def temperature_c(intake):
    temp = intake.get("temperature")
    if not temp:
        return None
    return temp["value"] if temp["unit"] == "C" else (temp["value"] - 32) * 5 / 9


def duration_days(intake):
    onset = intake.get("onset", {})
    value, unit = onset.get("duration_value"), onset.get("duration_unit")
    if value is None or unit is None:
        return None
    per_day = {"minutes": 1 / 1440, "hours": 1 / 24, "days": 1, "weeks": 7, "months": 30}
    return value * per_day[unit]


def condition_holds(condition, intake, rule):
    """True if every key in an `if` block holds. See `conditions` in the YAML.
    A condition about an unanswered field is false (except temperature_unknown)."""
    for key, want in condition.items():
        if key == "any_of":
            ok = any(condition_holds(c, intake, rule) for c in want)
        elif key == "red_flags":
            ok = bool(set(want) & set(intake.get("red_flags", [])))
        elif key == "signs":
            ok = bool(set(want) & set(intake.get("escalation_signs", [])))
        elif key in ("age_band", "trend"):
            ok = intake.get(key) in want
        elif key == "infant_age_months_min":
            ok = intake.get("infant_age_months") is not None and intake["infant_age_months"] >= want
        elif key == "infant_age_months_max":
            ok = intake.get("infant_age_months") is not None and intake["infant_age_months"] <= want
        elif key == "temperature_c_min":
            temp = temperature_c(intake)
            ok = temp is not None and temp >= want
        elif key == "temperature_unknown":
            ok = (temperature_c(intake) is None) == want
        elif key == "pregnant":
            ok = intake.get("pregnant") == want
        elif key == "chronic_conditions":
            ok = bool(set(want) & set(intake.get("chronic_conditions", [])))
        elif key == "duration_days_min":
            days = duration_days(intake)
            ok = days is not None and days >= want
        elif key == "beyond_expected_duration":
            days, limit = duration_days(intake), rule.get("expected_max_days")
            ok = (days is not None and limit is not None and days > limit) == want
        elif key == "no_modifiers":
            ok = (not any(condition_holds(m["if"], intake, rule)
                          for m in RULES["modifiers"])) == want
        else:
            raise ValueError(f"Unknown condition key in triage rules: {key}")
        if not ok:
            return False
    return True


# Intake fields each condition key reads, for reporting unanswered questions.
CONDITION_FIELDS = {
    "age_band": "age_band", "infant_age_months_min": "infant_age_months",
    "infant_age_months_max": "infant_age_months", "temperature_c_min": "temperature",
    "pregnant": "pregnant", "chronic_conditions": "chronic_conditions",
    "trend": "trend", "duration_days_min": "onset",
    "beyond_expected_duration": "onset",
}


def condition_fields(condition):
    for key, want in condition.items():
        if key == "any_of":
            for nested in want:
                yield from condition_fields(nested)
        elif key in CONDITION_FIELDS:
            yield CONDITION_FIELDS[key]


def missing_answers(rule, intake, level):
    """Unanswered intake fields that could raise the level if answered."""
    missing = []
    for line in rule.get("escalate_if", []):
        if rank(line["to"]) <= rank(level):
            continue
        band = line["if"].get("age_band")
        if band and intake.get("age_band") not in (None, "unknown") and intake["age_band"] not in band:
            continue  # rule is for another age group
        for field in condition_fields(line["if"]):
            if intake.get(field) in (None, "unknown", "not_asked", []) and field not in missing:
                missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Tool 3: find_care_nearby (Google Maps Platform, or fake data)
# ---------------------------------------------------------------------------

TRAVEL_MODES = {"car": "DRIVE", "rideshare": "DRIVE", "transit": "TRANSIT", "walking": "WALK"}


def google_request(url, body=None, field_mask=None):
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": MAPS_KEY}
    if field_mask:
        headers["X-Goog-FieldMask"] = field_mask
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # Name the API and pass on Google's own explanation (for example
        # "Places API (New) has not been used in project ... or it is disabled").
        try:
            message = json.loads(error.read())["error"]["message"]
        except Exception:
            message = str(error)
        api = urllib.parse.urlparse(url).netloc.split(".")[0]
        raise RuntimeError(f"Google {api} API error {error.code}: {message}") from None


def geocode(location_text):
    """City, ZIP or address -> (lat, lng) with the Geocoding API."""
    query = urllib.parse.urlencode({"address": location_text, "key": MAPS_KEY})
    with urllib.request.urlopen(
        f"https://maps.googleapis.com/maps/api/geocode/json?{query}", timeout=15
    ) as response:
        data = json.loads(response.read())
    if data.get("status") == "ZERO_RESULTS":
        raise ValueError("Could not find that location.")
    if data.get("status") != "OK":
        raise RuntimeError(f"Google Geocoding API error {data.get('status')}: "
                           f"{data.get('error_message', 'no details')}")
    point = data["results"][0]["geometry"]["location"]
    return point["lat"], point["lng"]


def search_places(query, lat, lng, radius_m, open_now):
    """Text search with Places API (New), biased to a circle around the person."""
    body = {
        "textQuery": query,
        "locationBias": {"circle": {
            "center": {"latitude": lat, "longitude": lng},
            "radius": min(radius_m, 50000.0),  # API maximum
        }},
        "maxResultCount": 8,
    }
    if open_now:
        body["openNow"] = True
    data = google_request(
        "https://places.googleapis.com/v1/places:searchText",
        body,
        "places.id,places.displayName,places.formattedAddress,places.location,"
        "places.nationalPhoneNumber,places.currentOpeningHours.openNow,places.googleMapsUri,"
        "places.primaryType,places.types",
    )
    return data.get("places", [])


def travel_times(lat, lng, places, mode):
    """Distance and travel time from the person to each place (Routes API)."""
    body = {
        "origins": [{"waypoint": {"location": {"latLng": {"latitude": lat, "longitude": lng}}}}],
        "destinations": [
            {"waypoint": {"location": {"latLng": p["location"]}}} for p in places
        ],
        "travelMode": mode,
    }
    rows = google_request(
        "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix",
        body,
        "originIndex,destinationIndex,distanceMeters,duration,condition",
    )
    return {row["destinationIndex"]: row for row in rows if row.get("condition") == "ROUTE_EXISTS"}


def straight_line_meters(lat, lng, point):
    """Distance as the crow flies, used when the Routes API is unavailable."""
    lat1, lng1 = math.radians(lat), math.radians(lng)
    lat2, lng2 = math.radians(point["latitude"]), math.radians(point["longitude"])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(a))


def is_emergency_room(place):
    """Hospitals and emergency departments. A text search for urgent care or a
    clinic often returns these too, so they are filtered by level of care."""
    types = set(place.get("types", [])) | {place.get("primaryType")}
    name = place.get("displayName", {}).get("text", "")
    return "hospital" in types or "general_hospital" in types or bool(
        re.search(r"\bemergency\b|\bER\b|\bhospital\b", name, re.IGNORECASE))


def search_level(level, lat, lng, radius_m, open_now, include_low_cost):
    """Places for one level of care. Emergency rooms count only for the er level."""
    queries = [LEVELS[level]["maps_search"]]
    if include_low_cost:
        queries.append("community health center free clinic")
    places, seen = [], set()
    for query in queries:
        for place in search_places(query, lat, lng, radius_m, open_now):
            if place["id"] in seen or is_emergency_room(place) != (level == "er"):
                continue
            seen.add(place["id"])
            place["low_cost"] = query != queries[0]
            place["level"] = level
            places.append(place)
    return places


def find_care_nearby(intake, level, open_now_only=False, include_low_cost=False):
    if level == "call_911":
        return {"error": "Do not search for call_911; tell the person to call 911."}
    location, radius = intake.get("location"), intake.get("travel_radius")
    if not location or not radius:
        return {"error": "Location and travel_radius are required first."}
    if not MAPS_KEY:
        return fake_results(level)

    lat, lng = geocode(location["text"])
    unit, value = radius["unit"], radius["value"]
    radius_m = value * MILE_IN_METERS if unit == "miles" else value * 1000
    if unit == "minutes":
        radius_m = 50000.0  # search wide, then filter by travel time below
    mode = TRAVEL_MODES.get(intake.get("transport"), "DRIVE")
    notes = []

    def results_for(places):
        if not places:
            return []
        try:
            routes = travel_times(lat, lng, places, mode)
        except Exception as error:  # still list places, with straight-line distance
            routes = None
            print(f"Routes API failed, using straight-line distances: {error}")
        results = []
        for i, place in enumerate(places):
            route = (routes or {}).get(i, {})
            meters = route.get("distanceMeters")
            if routes is None and place.get("location"):
                meters = straight_line_meters(lat, lng, place["location"])
            seconds = int(route.get("duration", "0s").rstrip("s") or 0)
            if unit == "minutes":
                in_radius = seconds and seconds / 60 <= value
            else:
                in_radius = meters is not None and meters <= radius_m
            results.append({
                "name": place["displayName"]["text"],
                "care_type": LEVELS[place["level"]]["label"],
                "address": place.get("formattedAddress"),
                "distance_miles": round(meters / MILE_IN_METERS, 1) if meters else None,
                "travel_time_minutes": round(seconds / 60) if seconds else None,
                "open_now": place.get("currentOpeningHours", {}).get("openNow"),
                "phone": place.get("nationalPhoneNumber"),
                "in_radius": bool(in_radius),
                "low_cost": place["low_cost"],
                "maps_url": place.get("googleMapsUri"),
            })
        if routes is None and not any("straight-line" in n for n in notes):
            notes.append("Travel times are unavailable; distances are straight-line, "
                         "so say 'about N miles away' and give no drive times.")
        return results

    results = results_for(search_level(level, lat, lng, radius_m, open_now_only, include_low_cost))

    # Nothing at the suggested level within the radius: offer the nearest higher
    # level instead (for example an ER when no urgent care is close), and say so.
    if not any(r["in_radius"] for r in results):
        higher = sorted((l for l in LEVELS.values() if l["maps_search"] and l["rank"] > LEVELS[level]["rank"]),
                        key=lambda l: l["rank"])
        for step in higher:
            extra = results_for(search_level(step["id"], lat, lng, radius_m, open_now_only, False))
            if any(r["in_radius"] for r in extra):
                notes.append(f"No {LEVELS[level]['label'].lower()} was found within the person's "
                             f"radius, so these include the nearest {step['label'].lower()} options. "
                             "Say that plainly, and that these places can also help.")
                results += extra
                break

    results.sort(key=lambda r: (not r["in_radius"], r["distance_miles"] or 9999))
    answer = {"results": results}
    if notes:
        answer["note"] = " ".join(notes)
    return answer


def fake_results(level):
    """Made-up places for testing without a Maps key. Never shown as real."""
    label = LEVELS[level]["label"]
    return {"fake_data": True, "results": [
        {"name": f"Test {label} A", "care_type": label, "distance_miles": 1.2,
         "travel_time_minutes": 6, "open_now": True, "phone": "(555) 555-0101",
         "in_radius": True, "low_cost": False},
        {"name": f"Test {label} B", "care_type": label, "distance_miles": 12.5,
         "travel_time_minutes": 25, "open_now": False, "phone": "(555) 555-0102",
         "in_radius": False, "low_cost": True},
    ]}


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_tool(name, tool_input, intake):
    if name == "record_intake":
        return record_intake(intake, tool_input)
    if name == "get_care_level":
        result = get_care_level(intake)
        # TODO: log result["rule_id"] and level (no personal data) per the build plan.
        return result
    if name == "find_care_nearby":
        return find_care_nearby(intake, **tool_input)
    if name == "offer_choices":
        return {"shown": True}  # the web page draws the chips; nothing to compute
    return {"error": f"Unknown tool {name}"}


def add_user_text(messages, text):
    """Add the person's message. A turn that ended on offer_choices leaves its tool
    result as the last user message, so the new text joins that message."""
    last = messages[-1] if messages else None
    if last and last["role"] == "user" and isinstance(last["content"], list):
        last["content"].append({"type": "text", "text": text})
    else:
        messages.append({"role": "user", "content": text})


def only_choices(response):
    """True when Claude's only tool call this round is offer_choices, so its reply is done."""
    calls = [b for b in response.content if b.type == "tool_use"]
    return bool(calls) and all(b.name == "offer_choices" for b in calls)


def chat():
    client = anthropic.Anthropic()
    system_prompt = PROMPT_PATH.read_text()
    tools = load_tools()
    messages, intake = [], {}
    print("Angie, from TriAgent (type 'quit' to stop)."
          + ("" if MAPS_KEY else " No MAPS_API_KEY set: using fake clinics."))

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        add_user_text(messages, text)

        # Keep calling Claude until it answers with text instead of tool calls.
        while True:
            response = client.messages.create(
                model=MODEL, max_tokens=1500, system=system_prompt,
                tools=tools, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\nAngie: {block.text}")
            if response.stop_reason != "tool_use":
                break

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    output = run_tool(block.name, block.input, intake)
                    is_error = "error" in output
                except Exception as error:  # report tool failures back to Claude
                    output, is_error = {"error": str(error)}, True
                if block.name == "offer_choices":
                    question = block.input.get("question", "")
                    said = " ".join(b.text for b in response.content if b.type == "text")
                    if question and question not in said:
                        print(f"\nAngie: {question}")
                    print("  Options: " + " · ".join(block.input.get("options", [])))
                if os.environ.get("TRIAGENT_DEBUG"):
                    print(f"  [{block.name}] {json.dumps(output)[:300]}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output), "is_error": is_error})
            messages.append({"role": "user", "content": results})
            if only_choices(response):
                break  # the question and its options are shown; wait for the person


if __name__ == "__main__":
    chat()
