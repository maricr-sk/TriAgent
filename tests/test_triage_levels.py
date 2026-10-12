"""Checks the urgency level for 20 sample cases. Run: python tests/test_triage_levels.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app")); sys.dont_write_bytecode=True
import triagent as t
S=lambda c,**k: {"symptoms":[{"description":c,"complaint_id":c}],**k}
cases=[
 ("mild hives", S("allergic_reaction"), "self_care"),
 ("spreading hives", S("allergic_reaction",escalation_signs=["spreading_hives_or_swelling"]), "urgent_care"),
 ("anaphylaxis", {"red_flags":["anaphylaxis"]}, "call_911"),
 ("infant 2mo 38.3C", S("fever",age_band="infant_under_1",infant_age_months=2,temperature={"value":100.9,"unit":"F"}), "urgent_care"),
 ("infant 2mo no temp", S("fever",age_band="infant_under_1",infant_age_months=2), "urgent_care"),
 ("infant 4mo 38.5C", S("fever",age_band="infant_under_1",infant_age_months=4,temperature={"value":38.5,"unit":"C"}), "clinic_or_telehealth"),
 ("infant 4mo 39.2C", S("fever",age_band="infant_under_1",infant_age_months=4,temperature={"value":39.2,"unit":"C"}), "urgent_care"),
 ("adult fever", S("fever",age_band="adult_18_to_64"), "self_care"),
 ("adult fever worse", S("fever",age_band="adult_18_to_64",trend="getting_worse"), "clinic_or_telehealth"),
 ("fever no meds work", S("fever",escalation_signs=["fever_not_responding_to_medicine"]), "er"),
 ("child fever 6 days", S("fever",age_band="child_1_to_17",onset={"duration_value":6,"duration_unit":"days"}), "clinic_or_telehealth"),
 ("chest pain", S("chest_pain"), "er"),
 ("chest comes+goes", S("chest_pain",trend="comes_and_goes"), "clinic_or_telehealth"),
 ("chest comes+goes 65+", S("chest_pain",trend="comes_and_goes",age_band="adult_65_plus"), "er"),
 ("stomach pregnant", S("abdominal_pain",pregnant="yes"), "urgent_care"),
 ("stomach vomit diabetes", S("abdominal_pain",escalation_signs=["vomiting"],chronic_conditions=["diabetes"]), "er"),
 ("stomach 4 days", S("abdominal_pain",onset={"duration_value":4,"duration_unit":"days"}), "clinic_or_telehealth"),
 ("deep cut", S("minor_injury_cut",escalation_signs=["large_or_deep_wound"]), "er"),
 ("dog bite stray", S("minor_injury_cut",escalation_signs=["bite_human_or_wild_animal"]), "urgent_care"),
 ("ear ache", S("other"), "clinic_or_telehealth"),
]
bad=0
for name,intake,want in cases:
    r=t.get_care_level(intake); ok=r['level']==want; bad+=not ok
    print(f"{'ok ' if ok else 'BAD'} {name:24} {r['level']:21} {r['rule_id']:28} missing={r.get('missing')}")
print(bad,"failures")
sys.exit(1 if bad else 0)
