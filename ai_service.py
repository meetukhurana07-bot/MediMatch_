"""
AI service layer using Google Gemini 2.5 Flash (google-genai SDK).
Provides intelligent match analysis, medical guidance, and conversational support.
"""

from google import genai
from google.genai import types
from typing import Optional

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

SYSTEM_PROMPT = """
You are MediMatch AI, an expert medical assistant specializing in blood and organ donation.
Your responsibilities:
1. Analyse donor-recipient compatibility with medical accuracy.
2. Provide empathetic, clear guidance to patients, donors, and hospital staff.
3. Explain blood-type and organ compatibility rules in simple language.
4. Flag urgent situations and suggest immediate actions.
5. Always recommend consulting a qualified transplant physician for final decisions.

Guidelines:
- Be empathetic and compassionate — these are life-or-death situations.
- Use plain, accessible language (avoid overly technical jargon).
- Cite blood type rules accurately (ABO, Rh factor).
- For organ matches, mention key factors: blood type, age, distance, urgency.
- Remind users that AI analysis is advisory only and not a medical diagnosis.
- If asked about unrelated topics, gently redirect to donation/medical context.
"""


def _make_client(api_key: str) -> genai.Client:
    """Return a configured google-genai Client."""
    return genai.Client(api_key=api_key)


def analyse_match(
    api_key: str,
    donor: dict,
    request: dict,
    compatibility_result: dict,
) -> str:
    """
    Generate an AI narrative analysis for a specific donor-recipient pair.
    Returns markdown-formatted text.
    """
    dist_str = (
        f"{compatibility_result['distance_km']:.0f} km"
        if compatibility_result.get("distance_km") is not None
        else "unknown"
    )
    breakdown = compatibility_result.get("breakdown", {})
    notes = compatibility_result.get("notes", [])

    prompt = f"""
Analyse this potential donation match and provide a concise medical assessment:

**DONOR PROFILE**
- Name: {donor.get('name')}
- Age: {donor.get('age')} years
- Blood Type: {donor.get('blood_type')}
- Location: {donor.get('city')}, {donor.get('state')}
- Offerings: {donor.get('donation_types')}
- Medical Notes: {donor.get('medical_notes') or 'None provided'}
- Currently Available: {donor.get('is_available')}

**PATIENT / RECIPIENT PROFILE**
- Patient: {request.get('patient_name')}, Age {request.get('age')}
- Blood Type: {request.get('blood_type')}
- Required: {request.get('required_donation')}
- Hospital: {request.get('hospital_name')}, {request.get('city')}, {request.get('state')}
- Urgency: {request.get('urgency_level')}
- Medical Description: {request.get('medical_description') or 'Not provided'}

**COMPATIBILITY RESULTS**
- Overall Score: {compatibility_result['score']}/100
- Distance: {dist_str}
- Blood/Organ Score: {breakdown.get('blood_organ', 0)}/40
- Location Score: {breakdown.get('location', 0)}/20
- Availability Score: {breakdown.get('availability', 0)}/10
- Urgency Bonus: {breakdown.get('urgency_bonus', 0)}/10
- Notes: {', '.join(notes) if notes else 'None'}

Please provide:
1. **Match Summary** – Is this a good match? Why?
2. **Medical Considerations** – Key factors and any concerns.
3. **Recommended Actions** – What should the hospital/coordinator do next?
4. **Urgency Assessment** – How quickly should this be acted upon?

Keep the response professional but accessible. Use bullet points for clarity.
"""
    try:
        client = _make_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI analysis unavailable: {str(e)}"


def chat_with_ai(
    api_key: str,
    user_message: str,
    chat_history: list[dict],
    context: Optional[str] = None,
) -> str:
    """
    Multi-turn conversational AI for donor/recipient guidance.
    chat_history: list of {"role": "user"|"model", "parts": [str]}
    """
    try:
        client = _make_client(api_key)
        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            ),
            history=[
                types.Content(role=msg["role"], parts=[types.Part(text=msg["parts"][0])])
                for msg in chat_history
            ],
        )
        full_message = user_message
        if context:
            full_message = f"[Context: {context}]\n\nUser question: {user_message}"

        response = chat.send_message(full_message)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to get AI response: {str(e)}"


def generate_urgency_alert(api_key: str, request: dict) -> str:
    """Generate a short urgent broadcast message for a critical request."""
    prompt = f"""
Write a concise, urgent appeal (max 100 words) for blood/organ donation social sharing:

Patient: {request.get('patient_name')}, Age {request.get('age')}
Needs: {request.get('required_donation')} — Blood Type: {request.get('blood_type')}
Hospital: {request.get('hospital_name')}, {request.get('city')}
Urgency: {request.get('urgency_level')}
Contact: {request.get('contact_name')} — {request.get('contact_phone')}

Make it compassionate, factual, and include a clear call-to-action.
Do NOT include personal identifiers beyond what's given.
"""
    try:
        client = _make_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.8,
                max_output_tokens=200,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate alert: {str(e)}"


def suggest_nearest_hospitals(api_key: str, city: str, organ: str) -> str:
    """Ask Gemini for general guidance on transplant centres for a given organ/city."""
    prompt = f"""
A patient in {city} needs a {organ} transplant/donation.

Provide:
1. General guidance on how to find accredited transplant centres in India near {city}.
2. Key organisations to contact (e.g., NOTTO, ROTTO, state organ authority).
3. What documents/steps are typically needed for urgent organ donation registration.

Keep it practical and actionable. Note that specific hospital recommendations should 
be verified with official health authorities.
"""
    try:
        client = _make_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.5,
                max_output_tokens=1024,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ Could not retrieve guidance: {str(e)}"


# Fallback model chain — tried in order until one succeeds
_MODEL_FALLBACKS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


def _resolve_model(client) -> str:
    """
    Return the first model name from the fallback chain that the API accepts.
    Falls back to MODEL_NAME if none respond (e.g. no network at check time).
    """
    for name in _MODEL_FALLBACKS:
        try:
            client.models.generate_content(
                model=name,
                contents="Hi",
                config=types.GenerateContentConfig(max_output_tokens=3),
            )
            return name
        except Exception as e:
            if "NOT_FOUND" in str(e) or "404" in str(e) or "no longer available" in str(e):
                continue
            # Any other error (auth, quota) — model exists, stop here
            return name
    return MODEL_NAME


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Test whether the provided Gemini API key is valid."""
    if not api_key or len(api_key) < 20:
        return False, "API key appears too short."
    try:
        client = _make_client(api_key)
        resolved = _resolve_model(client)
        # Update the module-level MODEL_NAME so all subsequent calls use it
        import ai_service as _self
        _self.MODEL_NAME = resolved
        return True, f"API key is valid. Using model: {resolved}"
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid" in err.lower():
            return False, "Invalid API key. Please check your Gemini API key."
        if "quota" in err.lower():
            return False, "API key valid but quota exceeded."
        return False, f"Connection error: {err}"
