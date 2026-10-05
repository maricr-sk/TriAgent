# TriAgent
A care-navigation agent that helps people figure out where to go for a health concern. It offers suggestions only. It is not medical advice. In an emergency, call 911.

**How it works**
1. You'll describe the symptoms in plain language. Then you'll give a city and state, or zip code, plus how far you'll travel. 
2. TriAgent checks for danger signs, then sorts urgency into five levels: call 911, ER, urgent care, clinic or telehealth, or self-care.
3. It searches Google Maps for matching facilities within your radius, including community health centers and free clinics.
4. It replies with next steps and nearby options, clearly flagging anything past your radius.

**Built with:** the Claude API for intake, a YAML triage rules file based on NHS, MedlinePlus and CDC guidance, and Google Maps Platform (Places, Distance Matrix).

**Roadmap:** accessible UI, insurance-aware ranking, and a synthetic EHR (MockEHR) for history-aware triage.
