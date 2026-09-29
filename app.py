"""
MediMatch — Donor-Recipient Matching Platform
Single-file Streamlit app. API key is backend-only (secrets/.env).
"""

import re
import math
import os
import io
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from google import genai
from google.genai import types

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — API KEY (never exposed to frontend)
# ══════════════════════════════════════════════════════════════════════════════

def _load_api_key() -> str:
    """Load Gemini API key from backend sources only. Never shown in UI."""
    try:
        val = st.secrets.get("GEMINI_API_KEY", "")
        if val and str(val).strip() not in ("", "your_api_key_here"):
            return str(val).strip()
    except Exception:
        pass
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY") and "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and val not in ("your_api_key_here", ""):
                            return val
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# SAFETY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_ALLOWED_NAME    = re.compile(r"^[A-Za-z\s\.\-\']{2,100}$")
_ALLOWED_PHONE   = re.compile(r"^[\+\d\s\-\(\)]{7,20}$")
_ALLOWED_EMAIL   = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_TEXT        = 1000
_RATE_WINDOW_SEC = 60
_RATE_MAX_AI     = 8


def _sanitize(text: str, max_len: int = _MAX_TEXT) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    return text.strip()[:max_len]


def _validate_name(v: str) -> str | None:
    v = _sanitize(v, 100)
    if not v:
        return "Name is required."
    if not _ALLOWED_NAME.match(v):
        return "Name must contain only letters, spaces, hyphens or apostrophes (2-100 chars)."
    return None


def _validate_email(v: str) -> str | None:
    v = _sanitize(v, 150)
    if not v:
        return "Email is required."
    if not _ALLOWED_EMAIL.match(v):
        return "Enter a valid email address."
    return None


def _validate_phone(v: str) -> str | None:
    v = _sanitize(v, 20)
    if not v:
        return "Phone number is required."
    if not _ALLOWED_PHONE.match(v):
        return "Phone must be digits, spaces, +, -, ( ) only (7-20 chars)."
    return None


def _rate_check_ai() -> bool:
    now = time.time()
    calls = st.session_state.get("_ai_calls", [])
    calls = [t for t in calls if now - t < _RATE_WINDOW_SEC]
    if len(calls) >= _RATE_MAX_AI:
        return False
    calls.append(now)
    st.session_state["_ai_calls"] = calls
    return True


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

DATABASE_URL = "sqlite:///donation_platform.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Donor(Base):
    __tablename__ = "donors"
    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(100), nullable=False)
    age            = Column(Integer, nullable=False)
    blood_type     = Column(String(5), nullable=False)
    email          = Column(String(150), nullable=False)
    phone          = Column(String(20), nullable=False)
    city           = Column(String(100), nullable=False)
    state          = Column(String(100), nullable=False)
    country        = Column(String(100), nullable=False, default="India")
    latitude       = Column(Float, nullable=True)
    longitude      = Column(Float, nullable=True)
    donation_types = Column(Text, nullable=False)
    medical_notes  = Column(Text, nullable=True)
    is_available   = Column(Boolean, default=True)
    registered_at  = Column(DateTime, default=datetime.utcnow)


class UrgentRequest(Base):
    __tablename__       = "urgent_requests"
    id                  = Column(Integer, primary_key=True, index=True)
    patient_name        = Column(String(100), nullable=False)
    age                 = Column(Integer, nullable=False)
    blood_type          = Column(String(5), nullable=False)
    required_donation   = Column(String(50), nullable=False)
    hospital_name       = Column(String(200), nullable=False)
    city                = Column(String(100), nullable=False)
    state               = Column(String(100), nullable=False)
    country             = Column(String(100), nullable=False, default="India")
    latitude            = Column(Float, nullable=True)
    longitude           = Column(Float, nullable=True)
    urgency_level       = Column(String(20), nullable=False, default="High")
    contact_name        = Column(String(100), nullable=False)
    contact_phone       = Column(String(20), nullable=False)
    contact_email       = Column(String(150), nullable=False)
    medical_description = Column(Text, nullable=True)
    is_fulfilled        = Column(Boolean, default=False)
    created_at          = Column(DateTime, default=datetime.utcnow)
    deadline            = Column(DateTime, nullable=True)


class Match(Base):
    __tablename__       = "matches"
    id                  = Column(Integer, primary_key=True, index=True)
    donor_id            = Column(Integer, nullable=False)
    request_id          = Column(Integer, nullable=False)
    compatibility_score = Column(Float, nullable=False)
    distance_km         = Column(Float, nullable=True)
    status              = Column(String(20), default="Pending")
    ai_analysis         = Column(Text, nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)


def get_session():
    return SessionLocal()


@st.cache_resource
def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if db.query(Donor).count() == 0:
        sample_donors = [
            Donor(name="Arun Kumar",   age=28, blood_type="O+", email="arun@example.com",
                  phone="+91-9876543210", city="Mumbai",    state="Maharashtra",
                  latitude=19.0760, longitude=72.8777, donation_types="Blood,Kidney",
                  is_available=True, medical_notes="Healthy, non-smoker"),
            Donor(name="Priya Sharma", age=34, blood_type="A+", email="priya@example.com",
                  phone="+91-9876543211", city="Delhi",     state="Delhi",
                  latitude=28.6139, longitude=77.2090, donation_types="Blood,Cornea,Bone Marrow",
                  is_available=True),
            Donor(name="Ravi Singh",   age=45, blood_type="B+", email="ravi@example.com",
                  phone="+91-9876543212", city="Bangalore", state="Karnataka",
                  latitude=12.9716, longitude=77.5946, donation_types="Blood,Liver",
                  is_available=True),
            Donor(name="Meena Reddy",  age=29, blood_type="AB+", email="meena@example.com",
                  phone="+91-9876543213", city="Chennai",   state="Tamil Nadu",
                  latitude=13.0827, longitude=80.2707, donation_types="Blood,Kidney,Pancreas",
                  is_available=True),
            Donor(name="Kiran Patel",  age=38, blood_type="O-", email="kiran@example.com",
                  phone="+91-9876543214", city="Ahmedabad", state="Gujarat",
                  latitude=23.0225, longitude=72.5714, donation_types="Blood,Skin,Cornea",
                  is_available=True, medical_notes="Universal donor"),
        ]
        sample_requests = [
            UrgentRequest(patient_name="Suresh Nair", age=52, blood_type="O+",
                          required_donation="Kidney", hospital_name="Apollo Hospital",
                          city="Mumbai", state="Maharashtra", latitude=19.1136, longitude=72.8697,
                          urgency_level="Critical", contact_name="Dr. Sharma",
                          contact_phone="+91-9000001111", contact_email="apollo@example.com",
                          medical_description="End-stage renal disease, dialysis 3x/week"),
            UrgentRequest(patient_name="Lalitha Devi", age=41, blood_type="A+",
                          required_donation="Blood", hospital_name="AIIMS Delhi",
                          city="Delhi", state="Delhi", latitude=28.5672, longitude=77.2100,
                          urgency_level="High", contact_name="Dr. Verma",
                          contact_phone="+91-9000002222", contact_email="aiims@example.com",
                          medical_description="Post-surgery anemia, urgent transfusion needed"),
        ]
        db.add_all(sample_donors + sample_requests)
        db.commit()
    db.close()


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BLOOD_DONATE_TO = {
    "O-":  ["O-","O+","A-","A+","B-","B+","AB-","AB+"],
    "O+":  ["O+","A+","B+","AB+"],
    "A-":  ["A-","A+","AB-","AB+"],
    "A+":  ["A+","AB+"],
    "B-":  ["B-","B+","AB-","AB+"],
    "B+":  ["B+","AB+"],
    "AB-": ["AB-","AB+"],
    "AB+": ["AB+"],
}
BLOOD_RECEIVE_FROM = {
    "AB+": ["O-","O+","A-","A+","B-","B+","AB-","AB+"],
    "AB-": ["O-","A-","B-","AB-"],
    "A+":  ["O-","O+","A-","A+"],
    "A-":  ["O-","A-"],
    "B+":  ["O-","O+","B-","B+"],
    "B-":  ["O-","B-"],
    "O+":  ["O-","O+"],
    "O-":  ["O-"],
}
ORGAN_AGE_LIMITS = {
    "Kidney":      {"donor_max":70,"donor_min":18},
    "Liver":       {"donor_max":65,"donor_min":18},
    "Heart":       {"donor_max":55,"donor_min":18},
    "Lungs":       {"donor_max":55,"donor_min":18},
    "Pancreas":    {"donor_max":50,"donor_min":18},
    "Cornea":      {"donor_max":80,"donor_min":2},
    "Bone Marrow": {"donor_max":60,"donor_min":18},
    "Skin":        {"donor_max":65,"donor_min":18},
    "Small Intestine": {"donor_max":60,"donor_min":18},
    "Blood":       {"donor_max":65,"donor_min":18},
}
STRICT_BLOOD_ORGANS = {"Heart","Lungs","Pancreas","Small Intestine"}

CITY_COORDS = {
    "Mumbai":       (19.0760, 72.8777),
    "Delhi":        (28.6139, 77.2090),
    "Bangalore":    (12.9716, 77.5946),
    "Chennai":      (13.0827, 80.2707),
    "Kolkata":      (22.5726, 88.3639),
    "Hyderabad":    (17.3850, 78.4867),
    "Ahmedabad":    (23.0225, 72.5714),
    "Pune":         (18.5204, 73.8567),
    "Jaipur":       (26.9124, 75.7873),
    "Lucknow":      (26.8467, 80.9462),
    "Surat":        (21.1702, 72.8311),
    "Kanpur":       (26.4499, 80.3319),
    "Nagpur":       (21.1458, 79.0882),
    "Patna":        (25.5941, 85.1376),
    "Indore":       (22.7196, 75.8577),
    "Bhopal":       (23.2599, 77.4126),
    "Visakhapatnam":(17.6868, 83.2185),
    "Vadodara":     (22.3072, 73.1812),
    "Coimbatore":   (11.0168, 76.9558),
    "Kochi":        (9.9312,  76.2673),
    "Other":        (20.5937, 78.9629),
}
STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh",
    "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan",
    "Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
    "Delhi","Jammu & Kashmir","Ladakh","Puducherry",
]
BLOOD_TYPES    = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
ORGAN_TYPES    = ["Blood","Kidney","Liver","Heart","Lungs","Pancreas","Cornea","Bone Marrow","Skin","Small Intestine"]
URGENCY_LEVELS = ["Critical","High","Medium","Low"]
URGENCY_COLOUR = {"Critical":"#c0392b","High":"#e67e22","Medium":"#2980b9","Low":"#27ae60"}


# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def compute_score(donor_blood, donor_age, dlat, dlon, available, d_types,
                  rec_blood, organ, rlat, rlon, urgency="Medium"):
    if organ not in d_types:
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":[f"Donor does not offer {organ}"]}
    limits = ORGAN_AGE_LIMITS.get(organ, {})
    if not (limits.get("donor_min",18) <= donor_age <= limits.get("donor_max",70)):
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":["Donor age outside permitted range for this organ"]}
    if organ in STRICT_BLOOD_ORGANS and donor_blood != rec_blood:
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":[f"Exact blood type required for {organ}"]}
    if donor_blood not in BLOOD_RECEIVE_FROM.get(rec_blood, []):
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":[f"{donor_blood} is not compatible with recipient blood type {rec_blood}"]}
    organ_score = 40.0 if donor_blood == rec_blood else 30.0
    dist        = _haversine(dlat, dlon, rlat, rlon)
    loc_score   = (20 if dist is None else 20 if dist<50 else 15 if dist<200
                   else 10 if dist<500 else 5 if dist<1000 else 2)
    avail_score = 10.0 if available else 0.0
    urg_bonus   = {"Critical":10,"High":7,"Medium":4,"Low":1}.get(urgency, 4)
    score       = round(min((organ_score + loc_score + avail_score + urg_bonus) / 80 * 100, 100), 1)
    notes = []
    if donor_blood == rec_blood:
        notes.append("Exact blood type match")
    else:
        notes.append(f"{donor_blood} compatible with recipient type {rec_blood}")
    return {"score":score,"compatible":True,
            "breakdown":{"blood_organ":organ_score,"location":loc_score,
                         "availability":avail_score,"urgency_bonus":urg_bonus},
            "distance_km":dist,"notes":notes}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: deadline countdown string
# ══════════════════════════════════════════════════════════════════════════════

def _deadline_str(deadline):
    if not deadline:
        return None
    delta = deadline - datetime.utcnow()
    days  = delta.days
    if days < 0:
        return "OVERDUE"
    if days == 0:
        hours = delta.seconds // 3600
        return f"{hours}h remaining"
    return f"{days}d remaining"


# ══════════════════════════════════════════════════════════════════════════════
# AI SERVICE  (api_key from backend _load_api_key(), never from UI)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_FALLBACKS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

_AI_MODEL = "gemini-3.7-flash"

_SYSTEM_PROMPT = (
    "You are MediMatch AI, a medical assistant specialising in blood and organ donation. "
    "Be empathetic and concise. Use plain language. Cite ABO/Rh rules accurately. "
    "Always state that your analysis is advisory and final decisions require a qualified physician."
)


def _make_client():
    key = _load_api_key()
    if not key:
        raise ValueError("No Gemini API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml.")
    return genai.Client(api_key=key)


def _cfg(**kw):
    return types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT, **kw)


def _resolve_model() -> str:
    global _AI_MODEL
    try:
        c = _make_client()
        for name in _MODEL_FALLBACKS:
            try:
                c.models.generate_content(model=name, contents="Hi",
                                           config=types.GenerateContentConfig(max_output_tokens=3))
                _AI_MODEL = name
                return name
            except Exception as e:
                if any(x in str(e) for x in ["NOT_FOUND","404","no longer available","deprecated"]):
                    continue
                _AI_MODEL = name
                return name
    except Exception:
        pass
    return _AI_MODEL


def _ai_ready() -> bool:
    return bool(_load_api_key())


def ai_analyse_match(donor, request, result):
    if not _rate_check_ai():
        return "Rate limit reached. Please wait a moment before generating another analysis."
    dist_str = f"{result['distance_km']:.0f} km" if result.get("distance_km") else "unknown"
    bd = result.get("breakdown", {})
    prompt = (
        f"Analyse this donation match:\n"
        f"DONOR: {donor['name']}, Age {donor['age']}, Blood {donor['blood_type']}, "
        f"{donor['city']}, Offers: {donor['donation_types']}\n"
        f"PATIENT: {request['patient_name']}, Age {request['age']}, "
        f"Blood {request['blood_type']}, Needs: {request['required_donation']}\n"
        f"HOSPITAL: {request['hospital_name']}, {request['city']} | Urgency: {request['urgency_level']}\n"
        f"SCORE: {result['score']}/100 | Distance: {dist_str}\n"
        f"Blood/Organ: {bd.get('blood_organ',0)}/40 | Location: {bd.get('location',0)}/20\n\n"
        "Provide: 1) Match Summary 2) Medical Considerations 3) Recommended Actions 4) Urgency Assessment"
    )
    try:
        r = _make_client().models.generate_content(
            model=_resolve_model(), contents=prompt,
            config=_cfg(temperature=0.7, max_output_tokens=2048))
        return r.text
    except Exception as e:
        return f"AI analysis unavailable: {e}"


def ai_chat(user_msg, history, context=None):
    if not _rate_check_ai():
        return "Rate limit reached. Please wait a moment before sending another message."

    client = _make_client()
    msg = f"[Context: {context}]\n\n{user_msg}" if context else user_msg

    models_to_try = [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]

    last_error = None

    for model in models_to_try:
        try:
            chat = client.chats.create(
                model=model,
                config=_cfg(temperature=0.7, max_output_tokens=2048),
                history=[
                    types.Content(
                        role=m["role"],
                        parts=[types.Part(text=m["parts"][0])]
                    )
                    for m in history
                ]
            )

            return chat.send_message(msg).text

        except Exception as e:
            last_error = e

            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue

            return f"AI error: {e}"

    return f"AI error: All available Gemini models are temporarily unavailable. Last error: {last_error}"


def ai_broadcast(req):
    if not _rate_check_ai():
        return "Rate limit reached."
    prompt = (
        f"Write a compassionate urgent appeal (max 80 words) for donation sharing.\n"
        f"Patient: {req.get('patient_name')}, Age {req.get('age')} | "
        f"Needs: {req.get('required_donation')} ({req.get('blood_type')})\n"
        f"Hospital: {req.get('hospital_name')}, {req.get('city')} | "
        f"Urgency: {req.get('urgency_level')}\n"
        f"Contact: {req.get('contact_name')} — {req.get('contact_phone')}\n"
        "Include a clear call-to-action."
    )
    try:
        r = _make_client().models.generate_content(
            model=_resolve_model(), contents=prompt,
            config=_cfg(temperature=0.8, max_output_tokens=200))
        return r.text
    except Exception as e:
        return f"Error: {e}"


def ai_hospital_guide(city, organ):
    if not _rate_check_ai():
        return "Rate limit reached."
    prompt = (
        f"A patient in {city} needs a {organ} donation or transplant.\n"
        "Provide: 1) How to find accredited transplant centres nearby "
        "2) Key organisations to contact (NOTTO, ROTTO, state authority) "
        "3) Required documents and next steps."
    )
    try:
        r = _make_client().models.generate_content(
            model=_resolve_model(), contents=prompt,
            config=_cfg(temperature=0.5, max_output_tokens=1024))
        return r.text
    except Exception as e:
        return f"Error: {e}"


def ai_contact_sheet(req, matches):
    """Generate a plain-text contact sheet for the top matches."""
    if not _rate_check_ai():
        return "Rate limit reached."
    lines = [f"REQUEST: {req.required_donation} for {req.patient_name} ({req.blood_type}), "
             f"{req.hospital_name}, {req.city} — {req.urgency_level} urgency"]
    for i, m in enumerate(matches[:5], 1):
        d = m["donor"]
        r = m["result"]
        lines.append(f"  {i}. {d.name} | {d.blood_type} | {d.city} | "
                     f"Score {r['score']}/100 | {d.phone} | {d.email}")
    prompt = (
        "You are helping a hospital coordinator prepare an outreach contact sheet.\n"
        "Given the following matched donors for an urgent request, write a brief "
        "professional summary (max 150 words) explaining which donor to contact first "
        "and why, along with any key medical cautions.\n\n"
        + "\n".join(lines)
    )
    try:
        r = _make_client().models.generate_content(
            model=_resolve_model(), contents=prompt,
            config=_cfg(temperature=0.5, max_output_tokens=512))
        return r.text
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT — PAGE CONFIG & INIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediMatch — Blood & Organ Donation",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

for k, v in [("active_page","Home"),("chat_messages",[]),("_ai_calls",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

if "_model_resolved" not in st.session_state:
    if _ai_ready():
        _resolve_model()
    st.session_state["_model_resolved"] = True

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = ["Home", "Register as Donor", "Post Urgent Request",
             "Matching Dashboard", "Donor Map", "AI Assistant", "Manage Records"]

with st.sidebar:
    st.markdown("## MediMatch")
    st.caption("Blood & Organ Donation Platform")
    st.divider()

    selected = st.radio("Navigation", NAV_ITEMS,
                        index=NAV_ITEMS.index(st.session_state.active_page)
                        if st.session_state.active_page in NAV_ITEMS else 0,
                        label_visibility="collapsed")
    st.session_state.active_page = selected
    st.divider()

    db = get_session()
    _nd = db.query(Donor).filter(Donor.is_available==True).count()
    _nr = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).count()
    _nm = db.query(Match).count()
    db.close()

    st.markdown("**Platform Overview**")
    _s1, _s2, _s3 = st.columns(3)
    _s1.metric("Donors",   _nd)
    _s2.metric("Requests", _nr)
    _s3.metric("Matches",  _nm)
    st.divider()

    if _ai_ready():
        st.success("AI Assistant: Active")
    else:
        st.warning("AI Assistant: Configure GEMINI_API_KEY in secrets.toml")

    st.caption("Advisory tool only. All clinical decisions require a qualified physician.")

page = st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("MediMatch — Blood & Organ Donation Platform")
    st.markdown(
        "Connecting donors and recipients through AI-powered compatibility matching. "
        "Register as a donor, post urgent needs, and find the best match instantly."
    )

    db = get_session()
    td  = db.query(Donor).count()
    ad  = db.query(Donor).filter(Donor.is_available==True).count()
    orq = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).count()
    cr  = db.query(UrgentRequest).filter(
              UrgentRequest.is_fulfilled==False,
              UrgentRequest.urgency_level=="Critical").count()
    tm  = db.query(Match).count()
    db.close()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Donors",  td)
    c2.metric("Active Donors", ad)
    c3.metric("Open Requests", orq)
    c4.metric("Critical",      cr, delta_color="inverse")
    c5.metric("Matches Made",  tm)
    st.divider()

    db = get_session()
    try:
        donors_df = pd.read_sql(
            "SELECT blood_type, donation_types FROM donors WHERE is_available=1", db.bind)
        req_df = pd.read_sql(
            "SELECT urgency_level, required_donation FROM urgent_requests WHERE is_fulfilled=0", db.bind)
    finally:
        db.close()

    ca, cb, cc = st.columns(3)
    with ca:
        if not donors_df.empty:
            bt = donors_df["blood_type"].value_counts().reset_index()
            bt.columns = ["Blood Type","Count"]
            fig = px.pie(bt, names="Blood Type", values="Count",
                         title="Active Donors by Blood Type",
                         color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
            fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=280,
                               font=dict(family="sans-serif",size=13))
            st.plotly_chart(fig, use_container_width=True)
    with cb:
        if not req_df.empty:
            urg = req_df["urgency_level"].value_counts().reset_index()
            urg.columns = ["Urgency","Count"]
            fig = px.bar(urg, x="Urgency", y="Count", title="Open Requests by Urgency",
                         color="Urgency", color_discrete_map=URGENCY_COLOUR, text="Count")
            fig.update_layout(showlegend=False, margin=dict(t=40,b=0), height=280,
                               font=dict(family="sans-serif",size=13))
            st.plotly_chart(fig, use_container_width=True)
    with cc:
        if not donors_df.empty:
            rows_flat = [t.strip() for row in donors_df["donation_types"]
                         for t in str(row).split(",")]
            dt = pd.Series(rows_flat).value_counts().reset_index()
            dt.columns = ["Type","Count"]
            fig = px.bar(dt.head(8), x="Count", y="Type", title="Donors by Donation Type",
                         orientation="h", color="Count", color_continuous_scale="Reds")
            fig.update_layout(margin=dict(t=40,b=0), height=280,
                               coloraxis_showscale=False,
                               font=dict(family="sans-serif",size=13))
            st.plotly_chart(fig, use_container_width=True)

    # ── Supply vs Demand bar chart (new)
    st.divider()
    st.subheader("Supply vs Demand by Blood Type")
    db = get_session()
    try:
        sup_df = pd.read_sql(
            "SELECT blood_type FROM donors WHERE is_available=1", db.bind)
        dem_df = pd.read_sql(
            "SELECT blood_type FROM urgent_requests WHERE is_fulfilled=0", db.bind)
    finally:
        db.close()

    if not sup_df.empty or not dem_df.empty:
        sup_counts = sup_df["blood_type"].value_counts() if not sup_df.empty else pd.Series(dtype=int)
        dem_counts = dem_df["blood_type"].value_counts() if not dem_df.empty else pd.Series(dtype=int)
        all_bt = sorted(set(list(sup_counts.index) + list(dem_counts.index)))
        svd_fig = go.Figure()
        svd_fig.add_trace(go.Bar(
            name="Donors Available",
            x=all_bt, y=[sup_counts.get(bt,0) for bt in all_bt],
            marker_color="#2980b9"
        ))
        svd_fig.add_trace(go.Bar(
            name="Requests Open",
            x=all_bt, y=[dem_counts.get(bt,0) for bt in all_bt],
            marker_color="#c0392b"
        ))
        svd_fig.update_layout(
            barmode="group", height=280, margin=dict(t=10,b=30,l=0,r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="sans-serif", size=13)
        )
        st.plotly_chart(svd_fig, use_container_width=True)

    st.divider()

    # ── Critical alerts with deadline countdown (enhanced)
    db = get_session()
    critical_list = db.query(UrgentRequest).filter(
        UrgentRequest.is_fulfilled==False,
        UrgentRequest.urgency_level.in_(["Critical","High"])
    ).order_by(UrgentRequest.created_at.desc()).limit(5).all()
    db.close()

    if critical_list:
        st.subheader("Active High-Priority Requests")
        for req in critical_list:
            ddl = _deadline_str(req.deadline)
            ddl_tag = f" | Deadline: {ddl}" if ddl else ""
            with st.expander(
                f"[{req.urgency_level.upper()}] {req.required_donation} — "
                f"{req.patient_name} | {req.hospital_name}, {req.city}{ddl_tag}",
                expanded=(req.urgency_level=="Critical")
            ):
                r1,r2,r3 = st.columns(3)
                r1.markdown(
                    f"**Patient:** {req.patient_name}, Age {req.age}  \n"
                    f"**Blood Type:** `{req.blood_type}`")
                r2.markdown(
                    f"**Needs:** {req.required_donation}  \n"
                    f"**Hospital:** {req.hospital_name}")
                r3.markdown(
                    f"**Contact:** {req.contact_name}  \n"
                    f"**Phone:** {req.contact_phone}")
                if ddl:
                    if ddl == "OVERDUE":
                        st.error(f"Deadline status: {ddl}")
                    else:
                        st.warning(f"Deadline: {ddl}")
    else:
        st.info("No critical requests at this time.")

    st.divider()
    st.subheader("Quick Actions")
    q1,q2,q3,q4 = st.columns(4)
    with q1:
        if st.button("Register as Donor", use_container_width=True, type="primary"):
            st.session_state.active_page = "Register as Donor"; st.rerun()
    with q2:
        if st.button("Post Urgent Request", use_container_width=True, type="primary"):
            st.session_state.active_page = "Post Urgent Request"; st.rerun()
    with q3:
        if st.button("Find Matches", use_container_width=True):
            st.session_state.active_page = "Matching Dashboard"; st.rerun()
    with q4:
        if st.button("View Donor Map", use_container_width=True):
            st.session_state.active_page = "Donor Map"; st.rerun()

    st.divider()
    st.markdown("""
### How MediMatch Works

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Register** | Donors sign up with blood type, location and available organs |
| 2 | **Request** | Hospitals post urgent needs with full patient details |
| 3 | **Match** | The compatibility engine scores every donor against each request |
| 4 | **Analyse** | Gemini 2.5 Flash generates a detailed medical assessment |
| 5 | **Connect** | Hospital staff are provided with ranked, compatible donor contacts |
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER DONOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Register as Donor":
    st.title("Register as a Donor")
    st.markdown(
        "Join our network. Your registration could save up to **8 lives**. "
        "All information is stored securely and shared only with verified medical professionals."
    )

    edit_id = st.session_state.get("edit_donor_id")
    prefill = {}
    if edit_id:
        db = get_session()
        d = db.query(Donor).filter(Donor.id == edit_id).first()
        if d:
            prefill = {
                "name":d.name,"age":d.age,"blood_type":d.blood_type,"email":d.email,
                "phone":d.phone,"city":d.city,"state":d.state,
                "donation_types":d.donation_types.split(","),
                "medical_notes":d.medical_notes or "","is_available":d.is_available
            }
        db.close()
        st.info(f"Editing existing donor record — ID: {edit_id}")

    with st.form("donor_form", clear_on_submit=False):
        st.subheader("Personal Details")
        f1, f2 = st.columns(2)
        with f1:
            name       = st.text_input("Full Name *", value=prefill.get("name",""),
                                        placeholder="e.g. Arun Kumar")
            age        = st.number_input("Age *", min_value=18, max_value=80,
                                          value=int(prefill.get("age",25)))
            blood_type = st.selectbox("Blood Type *", BLOOD_TYPES,
                                       index=BLOOD_TYPES.index(prefill.get("blood_type","O+")))
        with f2:
            email        = st.text_input("Email Address *", value=prefill.get("email",""),
                                          placeholder="you@example.com")
            phone        = st.text_input("Phone Number *", value=prefill.get("phone",""),
                                          placeholder="+91-XXXXXXXXXX")
            is_available = st.checkbox("I am currently available to donate",
                                        value=prefill.get("is_available",True))

        st.subheader("Location")
        l1, l2 = st.columns(2)
        with l1:
            city_list = list(CITY_COORDS.keys())
            city = st.selectbox(
                "Nearest City *", city_list,
                index=city_list.index(prefill.get("city","Mumbai"))
                      if prefill.get("city") in city_list else 0)
        with l2:
            state = st.selectbox(
                "State *", STATES,
                index=STATES.index(prefill.get("state","Maharashtra"))
                      if prefill.get("state") in STATES else 0)

        st.subheader("Donation Preferences")
        st.caption("Select all donation types you are willing to offer.")
        pt = prefill.get("donation_types", ["Blood"])
        if isinstance(pt, str):
            pt = [x.strip() for x in pt.split(",")]
        organ_cols = st.columns(5)
        selected_types = []
        for i, organ in enumerate(ORGAN_TYPES):
            with organ_cols[i % 5]:
                if st.checkbox(organ, value=(organ in pt), key=f"org_{organ}"):
                    selected_types.append(organ)

        st.subheader("Medical Notes")
        medical_notes = st.text_area(
            "Additional information for the medical team (optional)",
            value=prefill.get("medical_notes",""), height=80, max_chars=_MAX_TEXT,
            placeholder="Any relevant medical history, current medications, etc."
        )

        st.markdown("---")
        consent = st.checkbox(
            "I confirm that all information provided is accurate and I give consent "
            "to be contacted by verified medical professionals for donation purposes. *"
        )
        submitted = st.form_submit_button("Register / Update Profile",
                                           type="primary", use_container_width=True)

    if submitted:
        errors = []
        err = _validate_name(name)
        if err: errors.append(err)
        err = _validate_email(email)
        if err: errors.append(err)
        err = _validate_phone(phone)
        if err: errors.append(err)
        if not selected_types: errors.append("Select at least one donation type.")
        if not consent:        errors.append("You must agree to the consent statement.")

        for e in errors: st.error(e)
        if not errors:
            lat, lon = CITY_COORDS.get(city, (20.5937, 78.9629))
            db = get_session()
            try:
                if edit_id:
                    d = db.query(Donor).filter(Donor.id == edit_id).first()
                    if d:
                        d.name=_sanitize(name,100); d.age=age; d.blood_type=blood_type
                        d.email=_sanitize(email,150); d.phone=_sanitize(phone,20)
                        d.city=city; d.state=state; d.latitude=lat; d.longitude=lon
                        d.donation_types=",".join(selected_types)
                        d.medical_notes=_sanitize(medical_notes) or None
                        d.is_available=is_available
                        db.commit()
                        st.success(f"Donor profile updated for {name}.")
                        st.session_state.edit_donor_id = None
                else:
                    nd = Donor(
                        name=_sanitize(name,100), age=age, blood_type=blood_type,
                        email=_sanitize(email,150), phone=_sanitize(phone,20),
                        city=city, state=state, country="India",
                        latitude=lat, longitude=lon,
                        donation_types=",".join(selected_types),
                        medical_notes=_sanitize(medical_notes) or None,
                        is_available=is_available
                    )
                    db.add(nd); db.commit(); db.refresh(nd)
                    st.success(
                        f"Thank you, {_sanitize(name,100)}. "
                        f"You are registered as donor ID {nd.id}.")
                    st.balloons()
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                db.close()

    st.divider()
    st.subheader("Registered Donors")
    db = get_session()
    donors = db.query(Donor).order_by(Donor.registered_at.desc()).all()
    db.close()
    if donors:
        rows = [{"ID":d.id,"Name":d.name,"Age":d.age,"Blood Type":d.blood_type,
                 "City":d.city,"Donation Types":d.donation_types,
                 "Available":"Yes" if d.is_available else "No",
                 "Registered":d.registered_at.strftime("%d %b %Y")} for d in donors]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No donors registered yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POST REQUEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Post Urgent Request":
    st.title("Post an Urgent Donation Request")
    st.markdown(
        "Submit a donation request on behalf of a patient. "
        "The system will immediately scan for compatible donors."
    )

    with st.expander("Urgency Level Reference"):
        u1,u2,u3,u4 = st.columns(4)
        u1.error("**Critical** — Life-threatening, hours matter")
        u2.warning("**High** — Days to act, condition deteriorating")
        u3.info("**Medium** — Weeks available, early action preferred")
        u4.success("**Low** — Scheduled procedure, ample time")

    with st.form("request_form", clear_on_submit=False):
        st.subheader("Patient Details")
        p1, p2 = st.columns(2)
        with p1:
            patient_name      = st.text_input("Patient Name *", placeholder="e.g. Suresh Nair")
            patient_age       = st.number_input("Patient Age *", min_value=0, max_value=120, value=35)
            blood_type        = st.selectbox("Patient Blood Type *", BLOOD_TYPES)
        with p2:
            required_donation = st.selectbox("Donation Required *", ORGAN_TYPES)
            urgency_level     = st.selectbox("Urgency Level *", URGENCY_LEVELS)
            deadline_days     = st.number_input(
                "Days until deadline (0 = no fixed deadline)",
                min_value=0, max_value=365, value=0)

        st.subheader("Hospital & Location")
        hospital_name = st.text_input("Hospital Name *", placeholder="e.g. Apollo Hospital")
        h1, h2 = st.columns(2)
        with h1:
            city_list = list(CITY_COORDS.keys())
            req_city  = st.selectbox("City *", city_list)
        with h2:
            req_state = st.selectbox("State *", STATES)

        st.subheader("Contact Information")
        ct1, ct2, ct3 = st.columns(3)
        with ct1: contact_name  = st.text_input("Contact Person *",
                                                  placeholder="Doctor / Family member")
        with ct2: contact_phone = st.text_input("Contact Phone *", placeholder="+91-XXXXXXXXXX")
        with ct3: contact_email = st.text_input("Contact Email *",
                                                  placeholder="doctor@hospital.com")

        medical_description = st.text_area(
            "Medical Description *", height=100, max_chars=_MAX_TEXT,
            placeholder="Describe the patient's condition and why the donation is needed..."
        )
        st.markdown("---")
        req_submitted = st.form_submit_button("Submit Request", type="primary",
                                               use_container_width=True)

    if req_submitted:
        errors = []
        err = _validate_name(patient_name)
        if err: errors.append(f"Patient name: {err}")
        if not _sanitize(hospital_name): errors.append("Hospital name is required.")
        err = _validate_name(contact_name)
        if err: errors.append(f"Contact name: {err}")
        err = _validate_phone(contact_phone)
        if err: errors.append(f"Contact phone: {err}")
        err = _validate_email(contact_email)
        if err: errors.append(f"Contact email: {err}")
        if not _sanitize(medical_description): errors.append("Medical description is required.")

        for e in errors: st.error(e)
        if not errors:
            lat, lon = CITY_COORDS.get(req_city, (20.5937, 78.9629))
            deadline = datetime.utcnow() + timedelta(days=deadline_days) if deadline_days > 0 else None
            db = get_session()
            try:
                new_req = UrgentRequest(
                    patient_name=_sanitize(patient_name,100), age=patient_age,
                    blood_type=blood_type, required_donation=required_donation,
                    hospital_name=_sanitize(hospital_name,200),
                    city=req_city, state=req_state, country="India",
                    latitude=lat, longitude=lon, urgency_level=urgency_level,
                    contact_name=_sanitize(contact_name,100),
                    contact_phone=_sanitize(contact_phone,20),
                    contact_email=_sanitize(contact_email,150),
                    medical_description=_sanitize(medical_description),
                    deadline=deadline
                )
                db.add(new_req); db.commit(); db.refresh(new_req)
                st.success(
                    f"Request #{new_req.id} submitted successfully. "
                    "Go to **Matching Dashboard** to find compatible donors."
                )
                if _ai_ready():
                    with st.spinner("Generating broadcast message..."):
                        alert = ai_broadcast({
                            "patient_name":patient_name,"age":patient_age,
                            "blood_type":blood_type,"required_donation":required_donation,
                            "hospital_name":hospital_name,"city":req_city,
                            "urgency_level":urgency_level,
                            "contact_name":contact_name,"contact_phone":contact_phone
                        })
                    st.subheader("AI Broadcast Message")
                    st.info(alert)
                    st.caption("Share this message to reach potential donors quickly.")
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                db.close()

    st.divider()
    st.subheader("Open Requests")
    db = get_session()
    open_reqs = db.query(UrgentRequest).filter(
        UrgentRequest.is_fulfilled==False
    ).order_by(UrgentRequest.created_at.desc()).all()
    db.close()
    if open_reqs:
        open_reqs.sort(key=lambda r: {"Critical":0,"High":1,"Medium":2,"Low":3}.get(r.urgency_level,99))
        for req in open_reqs:
            ddl = _deadline_str(req.deadline)
            ddl_tag = f" | {ddl}" if ddl else ""
            with st.expander(
                f"[{req.urgency_level}] {req.required_donation} — {req.patient_name} "
                f"({req.blood_type}) | {req.hospital_name}, {req.city}{ddl_tag}"
            ):
                r1,r2,r3 = st.columns(3)
                r1.markdown(
                    f"**ID:** #{req.id}  \n"
                    f"**Patient:** {req.patient_name}, Age {req.age}  \n"
                    f"**Blood Type:** `{req.blood_type}`")
                r2.markdown(
                    f"**Needs:** {req.required_donation}  \n"
                    f"**Hospital:** {req.hospital_name}  \n"
                    f"**Location:** {req.city}, {req.state}")
                r3.markdown(
                    f"**Contact:** {req.contact_name}  \n"
                    f"**Phone:** {req.contact_phone}  \n"
                    f"**Email:** {req.contact_email}")
                if ddl:
                    if ddl == "OVERDUE":
                        st.error(f"Deadline: {ddl}")
                    else:
                        st.warning(f"Deadline: {ddl}")
                if req.medical_description:
                    st.caption(req.medical_description)
    else:
        st.info("No open requests at this time.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MATCHING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Matching Dashboard":
    st.title("Matching Dashboard")
    st.markdown(
        "Run the compatibility engine to find ranked donors for each open request. "
        "AI analysis and a printable contact sheet are available for any match."
    )

    db = get_session()
    open_requests = db.query(UrgentRequest).filter(
        UrgentRequest.is_fulfilled==False
    ).order_by(UrgentRequest.created_at.desc()).all()
    all_donors = db.query(Donor).filter(Donor.is_available==True).all()
    db.close()

    if not open_requests:
        st.warning("No open requests found. Post an urgent request first.")
        if st.button("Post a Request"):
            st.session_state.active_page = "Post Urgent Request"; st.rerun()
    elif not all_donors:
        st.warning("No active donors registered.")
        if st.button("Register a Donor"):
            st.session_state.active_page = "Register as Donor"; st.rerun()
    else:
        with st.expander("Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1: filter_urgency = st.multiselect("Urgency Level", URGENCY_LEVELS,
                                                        default=URGENCY_LEVELS)
            with fc2: filter_organ   = st.multiselect("Donation Type", ORGAN_TYPES, default=[])
            with fc3: min_score      = st.slider("Minimum Compatibility Score", 0, 100, 30)

        filtered_reqs = [r for r in open_requests
                         if r.urgency_level in filter_urgency
                         and (not filter_organ or r.required_donation in filter_organ)]

        if not filtered_reqs:
            st.info("No requests match the selected filters.")
        else:
            labels = [
                f"#{r.id} | {r.urgency_level} | {r.required_donation} ({r.blood_type}) — "
                f"{r.patient_name} @ {r.hospital_name}, {r.city}"
                for r in filtered_reqs
            ]
            sel_label = st.selectbox("Select Request", labels)
            req       = filtered_reqs[labels.index(sel_label)]

            with st.container(border=True):
                ddl = _deadline_str(req.deadline)
                st.markdown(f"**Request #{req.id} — {req.urgency_level} Priority**"
                            + (f"  |  Deadline: **{ddl}**" if ddl else ""))
                rc1, rc2, rc3 = st.columns(3)
                rc1.markdown(
                    f"**Patient:** {req.patient_name}, Age {req.age}  \n"
                    f"**Blood Type:** `{req.blood_type}`")
                rc2.markdown(
                    f"**Needs:** {req.required_donation}  \n"
                    f"**Hospital:** {req.hospital_name}")
                rc3.markdown(
                    f"**Location:** {req.city}, {req.state}  \n"
                    f"**Contact:** {req.contact_name} | {req.contact_phone}")
                if req.medical_description:
                    st.caption(req.medical_description)

            if st.button("Run Compatibility Matching", type="primary", use_container_width=True):
                with st.spinner("Scanning all active donors..."):
                    results = []
                    for donor in all_donors:
                        d_types = [t.strip() for t in donor.donation_types.split(",")]
                        r = compute_score(
                            donor.blood_type, donor.age,
                            donor.latitude, donor.longitude,
                            donor.is_available, d_types,
                            req.blood_type, req.required_donation,
                            req.latitude, req.longitude, req.urgency_level
                        )
                        if r["compatible"] and r["score"] >= min_score:
                            results.append({"donor":donor,"result":r})
                    results.sort(key=lambda x: x["result"]["score"], reverse=True)
                    st.session_state[f"matches_{req.id}"] = results
                    db = get_session()
                    for m in results:
                        existing = db.query(Match).filter(
                            Match.donor_id==m["donor"].id,
                            Match.request_id==req.id).first()
                        if existing:
                            existing.compatibility_score = m["result"]["score"]
                        else:
                            db.add(Match(
                                donor_id=m["donor"].id, request_id=req.id,
                                compatibility_score=m["result"]["score"],
                                distance_km=m["result"]["distance_km"]))
                    db.commit(); db.close()

            matches = st.session_state.get(f"matches_{req.id}", [])
            if not matches:
                st.info("Click **Run Compatibility Matching** to find donors.")
            else:
                st.success(f"Found {len(matches)} compatible donor(s).")

                if len(matches) > 1:
                    names  = [m["donor"].name for m in matches[:10]]
                    scores = [m["result"]["score"] for m in matches[:10]]
                    fig = go.Figure(go.Bar(
                        x=scores, y=names, orientation="h",
                        marker_color=["#27ae60" if s>=70 else "#e67e22" if s>=50
                                      else "#2980b9" for s in scores],
                        text=[f"{s:.0f}" for s in scores], textposition="outside"
                    ))
                    fig.update_layout(
                        title="Donor Compatibility Scores", xaxis_title="Score (/100)",
                        yaxis={"autorange":"reversed"},
                        height=max(220, 36*len(matches[:10])),
                        margin=dict(l=120,r=60,t=40,b=30),
                        font=dict(family="sans-serif",size=13)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Contact sheet (new feature)
                if _ai_ready():
                    cs_key = f"cs_{req.id}"
                    if st.button("Generate AI Contact Sheet", key=f"cs_btn_{req.id}",
                                  use_container_width=True):
                        with st.spinner("Preparing contact sheet with Gemini 2.5 Flash..."):
                            sheet = ai_contact_sheet(req, matches)
                        st.session_state[cs_key] = sheet
                    if st.session_state.get(cs_key):
                        with st.expander("AI Contact Sheet", expanded=True):
                            st.markdown(st.session_state[cs_key])
                            # Build plain-text printable version
                            lines = [
                                f"MEDIMATCH CONTACT SHEET",
                                f"Generated: {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}",
                                f"",
                                f"REQUEST: {req.required_donation} for {req.patient_name} "
                                f"({req.blood_type})",
                                f"Hospital: {req.hospital_name}, {req.city}, {req.state}",
                                f"Urgency: {req.urgency_level}",
                                f"Contact: {req.contact_name} | {req.contact_phone} | "
                                f"{req.contact_email}",
                                f"",
                                f"TOP COMPATIBLE DONORS:",
                            ]
                            for i, m in enumerate(matches[:5], 1):
                                d = m["donor"]
                                r2 = m["result"]
                                dist = f"{r2['distance_km']:.0f} km" \
                                       if r2.get("distance_km") else "—"
                                lines.append(
                                    f"  {i}. {d.name} | {d.blood_type} | {d.city} | "
                                    f"Score {r2['score']}/100 | Dist {dist}")
                                lines.append(f"     Phone: {d.phone} | Email: {d.email}")
                            lines += ["", "AI SUMMARY:", st.session_state[cs_key]]
                            txt = "\n".join(lines)
                            st.download_button(
                                "Download Contact Sheet (.txt)", txt,
                                f"contact_sheet_req{req.id}.txt", "text/plain",
                                use_container_width=True)

                st.subheader("Detailed Match Results")
                for rank, mi in enumerate(matches, 1):
                    donor  = mi["donor"]
                    result = mi["result"]
                    score  = result["score"]
                    rating = ("Excellent" if score>=80 else "Good" if score>=60
                              else "Fair" if score>=40 else "Low")
                    with st.expander(
                        f"Rank {rank} — {rating} ({score}/100) | "
                        f"{donor.name} | {donor.blood_type} | {donor.city}",
                        expanded=(rank==1)
                    ):
                        mc1, mc2, mc3 = st.columns([2,2,1])
                        with mc1:
                            st.markdown(
                                f"**Donor:** {donor.name}, Age {donor.age}  \n"
                                f"**Blood Type:** `{donor.blood_type}`  \n"
                                f"**Location:** {donor.city}, {donor.state}  \n"
                                f"**Offers:** {donor.donation_types}  \n"
                                f"**Phone:** {donor.phone}  \n"
                                f"**Email:** {donor.email}"
                            )
                        with mc2:
                            bd = result.get("breakdown", {})
                            st.markdown("**Score Breakdown**")
                            st.progress(
                                int(bd.get("blood_organ",0)/40*100),
                                text=f"Blood / Organ compatibility: {bd.get('blood_organ',0):.0f} / 40")
                            st.progress(
                                int(bd.get("location",0)/20*100),
                                text=f"Location proximity: {bd.get('location',0):.0f} / 20")
                            st.progress(
                                int(bd.get("availability",0)/10*100),
                                text=f"Availability: {bd.get('availability',0):.0f} / 10")
                            st.progress(
                                int(bd.get("urgency_bonus",0)/10*100),
                                text=f"Urgency bonus: {bd.get('urgency_bonus',0):.0f} / 10")
                            if result.get("distance_km"):
                                st.metric("Distance", f"{result['distance_km']:.0f} km")
                            for note in result.get("notes",[]): st.caption(note)
                        with mc3:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number", value=score,
                                title={"text":"Score","font":{"size":12}},
                                gauge={
                                    "axis":{"range":[0,100]},
                                    "bar":{"color":"#2980b9"},
                                    "steps":[
                                        {"range":[0,40],"color":"#fadbd8"},
                                        {"range":[40,70],"color":"#fdebd0"},
                                        {"range":[70,100],"color":"#d5f5e3"}
                                    ]
                                },
                                number={"suffix":"/100"}
                            ))
                            gauge.update_layout(height=180,
                                                margin=dict(t=30,b=0,l=10,r=10))
                            st.plotly_chart(gauge, use_container_width=True,
                                            key=f"g_{req.id}_{donor.id}")

                        ai_key = f"ai_{req.id}_{donor.id}"
                        if _ai_ready():
                            if st.button("Generate AI Analysis",
                                          key=f"ai_btn_{req.id}_{donor.id}",
                                          use_container_width=True):
                                with st.spinner("Analysing match with Gemini 2.5 Flash..."):
                                    analysis = ai_analyse_match(
                                        {"name":donor.name,"age":donor.age,
                                         "blood_type":donor.blood_type,
                                         "city":donor.city,"state":donor.state,
                                         "donation_types":donor.donation_types,
                                         "medical_notes":donor.medical_notes,
                                         "is_available":donor.is_available},
                                        {"patient_name":req.patient_name,"age":req.age,
                                         "blood_type":req.blood_type,
                                         "required_donation":req.required_donation,
                                         "hospital_name":req.hospital_name,
                                         "city":req.city,"state":req.state,
                                         "urgency_level":req.urgency_level,
                                         "medical_description":req.medical_description},
                                        result
                                    )
                                st.session_state[ai_key] = analysis
                                db = get_session()
                                ex = db.query(Match).filter(
                                    Match.donor_id==donor.id,
                                    Match.request_id==req.id).first()
                                if ex:
                                    ex.ai_analysis = analysis; db.commit()
                                db.close()
                        else:
                            st.caption("AI analysis unavailable — configure GEMINI_API_KEY.")

                        if st.session_state.get(ai_key):
                            st.markdown("---")
                            st.markdown("**AI Medical Analysis**")
                            st.markdown(st.session_state[ai_key])

                st.divider()
                if st.button("Mark Request as Fulfilled", use_container_width=True):
                    db = get_session()
                    r2 = db.query(UrgentRequest).filter(UrgentRequest.id==req.id).first()
                    if r2: r2.is_fulfilled = True; db.commit()
                    db.close()
                    st.success(f"Request #{req.id} marked as fulfilled.")
                    st.session_state.pop(f"matches_{req.id}", None)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DONOR MAP  (new page)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Donor Map":
    st.title("Donor & Request Map")
    st.markdown(
        "Geographic view of all registered donors and open requests across India. "
        "Use the filters to narrow by blood type or donation type."
    )

    db = get_session()
    donors_all = db.query(Donor).filter(Donor.is_available==True).all()
    reqs_all   = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).all()
    db.close()

    mf1, mf2, mf3 = st.columns(3)
    with mf1:
        map_bt = st.multiselect("Blood Type", BLOOD_TYPES, default=[], key="map_bt")
    with mf2:
        map_organ = st.multiselect("Donation Type", ORGAN_TYPES, default=[], key="map_organ")
    with mf3:
        show_requests = st.toggle("Show open requests", value=True, key="map_req")

    # Filter donors
    d_filtered = donors_all
    if map_bt:
        d_filtered = [d for d in d_filtered if d.blood_type in map_bt]
    if map_organ:
        d_filtered = [d for d in d_filtered
                      if any(o in [t.strip() for t in d.donation_types.split(",")]
                             for o in map_organ)]
    d_filtered = [d for d in d_filtered if d.latitude and d.longitude]

    donor_df = pd.DataFrame([{
        "lat": d.latitude, "lon": d.longitude,
        "Name": d.name, "Blood Type": d.blood_type,
        "City": d.city, "Donation Types": d.donation_types,
        "Type": "Donor"
    } for d in d_filtered])

    req_df_map = pd.DataFrame([{
        "lat": r.latitude, "lon": r.longitude,
        "Name": f"{r.patient_name} ({r.required_donation})",
        "Blood Type": r.blood_type,
        "City": r.city, "Donation Types": r.required_donation,
        "Type": f"Request — {r.urgency_level}"
    } for r in reqs_all if r.latitude and r.longitude])

    frames = [donor_df]
    if show_requests and not req_df_map.empty:
        frames.append(req_df_map)
    map_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if map_df.empty:
        st.info("No geo-located donors match the selected filters.")
    else:
        color_map = {
            "Donor": "#2980b9",
            "Request — Critical": "#c0392b",
            "Request — High":     "#e67e22",
            "Request — Medium":   "#8e44ad",
            "Request — Low":      "#27ae60",
        }
        fig = px.scatter_geo(
            map_df,
            lat="lat", lon="lon",
            color="Type",
            color_discrete_map=color_map,
            hover_name="Name",
            hover_data={"Blood Type":True,"City":True,"Donation Types":True,
                        "lat":False,"lon":False},
            size_max=14,
            title="Donors (blue) and Open Requests (coloured by urgency)",
            scope="asia",
            fitbounds="locations",
        )
        fig.update_geos(
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=True, coastlinecolor="lightgrey",
            showland=True, landcolor="#f9f9f9",
            showocean=True, oceancolor="#eaf4fc",
        )
        fig.update_layout(
            height=520, margin=dict(t=40,b=0,l=0,r=0),
            font=dict(family="sans-serif",size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # City-level summary table
    st.subheader("Donors per City")
    if d_filtered:
        city_counts = {}
        for d in d_filtered:
            city_counts[d.city] = city_counts.get(d.city, 0) + 1
        city_df = pd.DataFrame(
            sorted(city_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["City","Available Donors"]
        )
        st.dataframe(city_df, use_container_width=True, hide_index=True)
    else:
        st.info("No donors to display.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Assistant":
    st.title("AI Medical Assistant")
    st.markdown(
        "Ask questions about blood type compatibility, organ donation rules, urgency assessment, "
        "and more. Powered by Gemini 2.5 Flash."
    )

    if not _ai_ready():
        st.warning(
            "The AI assistant is not configured. "
            "Add **GEMINI_API_KEY** to `.streamlit/secrets.toml` to enable this feature."
        )
    else:
        cfg1, cfg2 = st.columns([3,1])
        with cfg1:
            include_ctx = st.toggle("Include live platform statistics in AI context", value=True)
        with cfg2:
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.chat_messages = []; st.rerun()

        st.markdown("**Common Questions:**")
        qcols = st.columns(4)
        quick_prompts = [
            "What blood types are compatible with O+?",
            "Can a 65-year-old donate a kidney?",
            "What is the maximum transport time for a donor heart?",
            "How does bone marrow matching work?",
        ]
        quick_sel = None
        for i, qp in enumerate(quick_prompts):
            with qcols[i]:
                if st.button(qp, key=f"qp_{i}", use_container_width=True):
                    quick_sel = qp
        st.divider()

        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_messages:
                st.info(
                    "Hello — I am MediMatch AI. Ask me anything about blood types, "
                    "organ compatibility, donation processes, or urgent medical situations."
                )
            for msg in st.session_state.chat_messages:
                role   = msg["role"]
                avatar = "assistant" if role == "assistant" else "user"
                with st.chat_message(avatar):
                    st.markdown(msg["content"])

        user_input = st.chat_input("Type your question here...")
        if quick_sel:
            user_input = quick_sel

        if user_input:
            user_input = _sanitize(user_input, 500)
            if not user_input:
                st.warning("Please enter a valid message.")
            else:
                st.session_state.chat_messages.append({"role":"user","content":user_input})
                with chat_container:
                    with st.chat_message("user"): st.markdown(user_input)

                history = [
                    {"role":"user" if m["role"]=="user" else "model",
                     "parts":[m["content"]]}
                    for m in st.session_state.chat_messages[:-1]
                ]
                ctx = None
                if include_ctx:
                    db = get_session()
                    _nd2 = db.query(Donor).filter(Donor.is_available==True).count()
                    _nr2 = db.query(UrgentRequest).filter(
                               UrgentRequest.is_fulfilled==False).count()
                    _nc2 = db.query(UrgentRequest).filter(
                               UrgentRequest.is_fulfilled==False,
                               UrgentRequest.urgency_level=="Critical").count()
                    db.close()
                    ctx = f"{_nd2} active donors, {_nr2} open requests ({_nc2} critical)."

                with st.spinner("Thinking..."):
                    response = ai_chat(user_input, history, ctx)
                st.session_state.chat_messages.append({"role":"assistant","content":response})
                with chat_container:
                    with st.chat_message("assistant"): st.markdown(response)

        st.divider()
        with st.expander("Blood Type Compatibility Reference"):
            st.markdown("**Donor can give to:**")
            st.table({k:", ".join(v) for k,v in BLOOD_DONATE_TO.items()})
            st.markdown("**Recipient can receive from:**")
            st.table({k:", ".join(v) for k,v in BLOOD_RECEIVE_FROM.items()})

        with st.expander("Find Transplant Centre Guidance"):
            gc1, gc2 = st.columns(2)
            with gc1: guide_city  = st.text_input("City", placeholder="e.g. Mumbai")
            with gc2: guide_organ = st.selectbox("Organ / Donation Type", ORGAN_TYPES,
                                                  key="guide_organ")
            if st.button("Get Guidance", key="hosp_guide"):
                if guide_city:
                    with st.spinner("Querying AI..."):
                        st.markdown(ai_hospital_guide(_sanitize(guide_city,60), guide_organ))
                else:
                    st.warning("Please enter a city name.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MANAGE RECORDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Manage Records":
    st.title("Manage Records")
    st.markdown("View, edit, and manage donor profiles, donation requests, and match history.")

    with st.expander("Backup & Restore Data", expanded=False):
        st.markdown("**Export** your data as CSV, then **Import** it to restore after a cloud reset.")
        st.markdown("#### Cleanup")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Delete All Donors & Matches", use_container_width=True, type="primary"):
                db = get_session()
                db.query(Donor).delete(); db.query(Match).delete()
                db.commit(); db.close()
                st.success("All donor and match records deleted."); st.rerun()
        with cc2:
            if st.button("Delete All Requests & Matches", use_container_width=True, type="primary"):
                db = get_session()
                db.query(UrgentRequest).delete(); db.query(Match).delete()
                db.commit(); db.close()
                st.success("All request and match records deleted."); st.rerun()

        st.markdown("#### Export")
        db = get_session()
        _exp_donors = db.query(Donor).all()
        _exp_reqs   = db.query(UrgentRequest).all()
        db.close()
        ec1, ec2 = st.columns(2)
        with ec1:
            if _exp_donors:
                _d_rows = [{
                    "name":d.name,"age":d.age,"blood_type":d.blood_type,
                    "email":d.email,"phone":d.phone,"city":d.city,"state":d.state,
                    "donation_types":d.donation_types,
                    "medical_notes":d.medical_notes or "",
                    "is_available":d.is_available
                } for d in _exp_donors]
                st.download_button("Download Donors CSV",
                                   pd.DataFrame(_d_rows).to_csv(index=False),
                                   "donors_backup.csv", "text/csv", use_container_width=True)
        with ec2:
            if _exp_reqs:
                _r_rows = [{
                    "patient_name":r.patient_name,"age":r.age,"blood_type":r.blood_type,
                    "required_donation":r.required_donation,
                    "hospital_name":r.hospital_name,
                    "city":r.city,"state":r.state,"urgency_level":r.urgency_level,
                    "contact_name":r.contact_name,"contact_phone":r.contact_phone,
                    "contact_email":r.contact_email,
                    "medical_description":r.medical_description or ""
                } for r in _exp_reqs]
                st.download_button("Download Requests CSV",
                                   pd.DataFrame(_r_rows).to_csv(index=False),
                                   "requests_backup.csv", "text/csv", use_container_width=True)

        st.markdown("#### Import")
        imp_c1, imp_c2 = st.columns(2)
        with imp_c1:
            donors_file = st.file_uploader("Upload donors_backup.csv", type="csv",
                                            key="imp_donors")
            if donors_file and st.button("Import Donors", use_container_width=True,
                                          key="do_imp_donors"):
                try:
                    df_imp = pd.read_csv(io.StringIO(donors_file.read().decode("utf-8")))
                    db = get_session()
                    existing = {(d.email.strip().lower(), d.phone.strip())
                                for d in db.query(Donor).all()}
                    added = skipped = 0
                    for _, row in df_imp.iterrows():
                        key = (str(row["email"]).strip().lower(), str(row["phone"]).strip())
                        if key in existing: skipped += 1; continue
                        lat, lon = CITY_COORDS.get(str(row.get("city","")), (20.5937,78.9629))
                        db.add(Donor(
                            name=_sanitize(str(row["name"]),100), age=int(row["age"]),
                            blood_type=str(row["blood_type"]),
                            email=str(row["email"]).strip(),
                            phone=str(row["phone"]).strip(),
                            city=str(row["city"]), state=str(row["state"]),
                            country="India", latitude=lat, longitude=lon,
                            donation_types=str(row["donation_types"]),
                            medical_notes=_sanitize(str(row.get("medical_notes",""))) or None,
                            is_available=bool(row.get("is_available",True)),
                        ))
                        existing.add(key); added += 1
                    db.commit(); db.close()
                    msg = f"Imported {added} donor(s)."
                    if skipped: msg += f" Skipped {skipped} duplicate(s)."
                    st.success(msg); st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        with imp_c2:
            reqs_file = st.file_uploader("Upload requests_backup.csv", type="csv",
                                          key="imp_reqs")
            if reqs_file and st.button("Import Requests", use_container_width=True,
                                        key="do_imp_reqs"):
                try:
                    df_imp = pd.read_csv(io.StringIO(reqs_file.read().decode("utf-8")))
                    db = get_session()
                    existing = {
                        (r.patient_name.strip().lower(),
                         r.hospital_name.strip().lower(),
                         r.required_donation.strip().lower())
                        for r in db.query(UrgentRequest).all()
                    }
                    added = skipped = 0
                    for _, row in df_imp.iterrows():
                        key = (str(row["patient_name"]).strip().lower(),
                               str(row["hospital_name"]).strip().lower(),
                               str(row["required_donation"]).strip().lower())
                        if key in existing: skipped += 1; continue
                        lat, lon = CITY_COORDS.get(str(row.get("city","")), (20.5937,78.9629))
                        db.add(UrgentRequest(
                            patient_name=_sanitize(str(row["patient_name"]),100),
                            age=int(row["age"]),
                            blood_type=str(row["blood_type"]),
                            required_donation=str(row["required_donation"]),
                            hospital_name=_sanitize(str(row["hospital_name"]),200),
                            city=str(row["city"]), state=str(row["state"]),
                            country="India", latitude=lat, longitude=lon,
                            urgency_level=str(row.get("urgency_level","High")),
                            contact_name=_sanitize(str(row["contact_name"]),100),
                            contact_phone=_sanitize(str(row["contact_phone"]),20),
                            contact_email=_sanitize(str(row["contact_email"]),150),
                            medical_description=_sanitize(
                                str(row.get("medical_description",""))) or None,
                        ))
                        existing.add(key); added += 1
                    db.commit(); db.close()
                    msg = f"Imported {added} request(s)."
                    if skipped: msg += f" Skipped {skipped} duplicate(s)."
                    st.success(msg); st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    st.divider()
    tab_d, tab_r, tab_m = st.tabs(["Donors", "Requests", "Match History"])

    # ── Tab: Donors
    with tab_d:
        st.subheader("All Registered Donors")
        db = get_session()
        donors = db.query(Donor).order_by(Donor.registered_at.desc()).all()
        db.close()
        if not donors:
            st.info("No donors registered yet.")
        else:
            search = st.text_input("Search (name, city, blood type)", key="dsearch")
            bt_f   = st.multiselect("Blood Type Filter", BLOOD_TYPES, key="dbt")
            av_f   = st.radio("Availability", ["All","Available only","Unavailable only"],
                               horizontal=True, key="dav")
            filt = donors
            if search:
                s = search.lower()
                filt = [d for d in filt
                        if s in d.name.lower() or s in d.city.lower()
                        or s in d.blood_type.lower()]
            if bt_f: filt = [d for d in filt if d.blood_type in bt_f]
            if av_f == "Available only":   filt = [d for d in filt if d.is_available]
            elif av_f == "Unavailable only": filt = [d for d in filt if not d.is_available]

            st.caption(f"Showing {len(filt)} of {len(donors)} donors")
            for donor in filt:
                avail = "Available" if donor.is_available else "Unavailable"
                with st.expander(
                    f"{donor.name} | {donor.blood_type} | {donor.city} | ID: {donor.id}"
                ):
                    dc1, dc2, dc3 = st.columns(3)
                    dc1.markdown(
                        f"**Name:** {donor.name}  \n"
                        f"**Age:** {donor.age}  \n"
                        f"**Blood Type:** `{donor.blood_type}`")
                    dc2.markdown(
                        f"**Email:** {donor.email}  \n"
                        f"**Phone:** {donor.phone}  \n"
                        f"**Location:** {donor.city}, {donor.state}")
                    dc3.markdown(
                        f"**Donation Types:** {donor.donation_types}  \n"
                        f"**Status:** {avail}  \n"
                        f"**Registered:** {donor.registered_at.strftime('%d %b %Y')}")
                    if donor.medical_notes:
                        st.caption(f"Notes: {donor.medical_notes}")

                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("Edit", key=f"ed_{donor.id}", use_container_width=True):
                            st.session_state.edit_donor_id = donor.id
                            st.session_state.active_page = "Register as Donor"; st.rerun()
                    with b2:
                        lbl = "Set Available" if not donor.is_available else "Set Unavailable"
                        if st.button(lbl, key=f"av_{donor.id}", use_container_width=True):
                            db = get_session()
                            d = db.query(Donor).filter(Donor.id==donor.id).first()
                            if d: d.is_available = not donor.is_available; db.commit()
                            db.close(); st.rerun()
                    with b3:
                        if st.button("Delete", key=f"dd_{donor.id}", use_container_width=True):
                            db = get_session()
                            d = db.query(Donor).filter(Donor.id==donor.id).first()
                            if d: db.delete(d); db.commit()
                            db.close()
                            st.success(f"{donor.name} deleted."); st.rerun()

            if st.button("Export Donors as CSV", key="exp_d"):
                df = pd.DataFrame([{
                    "ID":d.id,"Name":d.name,"Age":d.age,"Blood Type":d.blood_type,
                    "City":d.city,"State":d.state,
                    "Donation Types":d.donation_types,"Available":d.is_available
                } for d in filt])
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "donors.csv", "text/csv")

    # ── Tab: Requests
    with tab_r:
        st.subheader("All Donation Requests")
        db = get_session()
        all_reqs = db.query(UrgentRequest).order_by(UrgentRequest.created_at.desc()).all()
        db.close()
        if not all_reqs:
            st.info("No requests posted yet.")
        else:
            show_f = st.toggle("Include fulfilled requests", value=False, key="show_ful")
            disp   = [r for r in all_reqs if show_f or not r.is_fulfilled]
            for req in disp:
                status = "Fulfilled" if req.is_fulfilled else "Open"
                ddl    = _deadline_str(req.deadline)
                ddl_tag = f" | {ddl}" if ddl else ""
                with st.expander(
                    f"[{req.urgency_level}] {req.required_donation} — "
                    f"{req.patient_name} | {status} | ID: {req.id}{ddl_tag}"
                ):
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.markdown(
                        f"**Patient:** {req.patient_name}, Age {req.age}  \n"
                        f"**Blood Type:** `{req.blood_type}`  \n"
                        f"**Needs:** {req.required_donation}")
                    rc2.markdown(
                        f"**Hospital:** {req.hospital_name}  \n"
                        f"**Location:** {req.city}, {req.state}  \n"
                        f"**Urgency:** {req.urgency_level}")
                    rc3.markdown(
                        f"**Contact:** {req.contact_name}  \n"
                        f"**Phone:** {req.contact_phone}  \n"
                        f"**Posted:** {req.created_at.strftime('%d %b %Y')}")
                    if ddl:
                        st.caption(f"Deadline: {ddl}")
                    if req.medical_description:
                        st.caption(req.medical_description)

                    rb1, rb2, rb3 = st.columns(3)
                    with rb1:
                        if not req.is_fulfilled:
                            if st.button("Mark Fulfilled", key=f"ful_{req.id}",
                                          use_container_width=True):
                                db = get_session()
                                r2 = db.query(UrgentRequest).filter(
                                    UrgentRequest.id==req.id).first()
                                if r2: r2.is_fulfilled = True; db.commit()
                                db.close(); st.rerun()
                    with rb2:
                        if st.button("Find Matches", key=f"mr_{req.id}",
                                      use_container_width=True):
                            st.session_state.active_page = "Matching Dashboard"; st.rerun()
                    with rb3:
                        if st.button("Delete", key=f"dr_{req.id}",
                                      use_container_width=True):
                            db = get_session()
                            r2 = db.query(UrgentRequest).filter(
                                UrgentRequest.id==req.id).first()
                            if r2: db.delete(r2); db.commit()
                            db.close()
                            st.success(f"Request #{req.id} deleted."); st.rerun()

    # ── Tab: Match History
    with tab_m:
        st.subheader("Match History")
        db = get_session()
        matches   = db.query(Match).order_by(Match.created_at.desc()).all()
        donor_map = {d.id:d for d in db.query(Donor).all()}
        req_map   = {r.id:r for r in db.query(UrgentRequest).all()}
        db.close()
        if not matches:
            st.info("No matches generated yet. Run the Matching Dashboard first.")
        else:
            rows = []
            for m in matches:
                d = donor_map.get(m.donor_id)
                r = req_map.get(m.request_id)
                rows.append({
                    "ID":       m.id,
                    "Donor":    d.name if d else f"ID {m.donor_id}",
                    "Blood":    d.blood_type if d else "?",
                    "Patient":  r.patient_name if r else f"ID {m.request_id}",
                    "Needs":    r.required_donation if r else "?",
                    "Score":    f"{m.compatibility_score:.1f}/100",
                    "Distance": f"{m.distance_km:.0f} km" if m.distance_km else "—",
                    "Status":   m.status,
                    "Date":     m.created_at.strftime("%d %b %Y"),
                    "AI":       "Yes" if m.ai_analysis else "No",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            ai_matches = [m for m in matches if m.ai_analysis]
            if ai_matches:
                sel_id = st.selectbox("View AI Analysis for Match ID",
                                       [m.id for m in ai_matches], key="vm")
                sel = next((m for m in ai_matches if m.id==sel_id), None)
                if sel:
                    d = donor_map.get(sel.donor_id)
                    r = req_map.get(sel.request_id)
                    st.markdown(
                        f"**{d.name if d else '?'}** → **{r.patient_name if r else '?'}** "
                        f"(Score: {sel.compatibility_score:.1f}/100)"
                    )
                    st.markdown(sel.ai_analysis)

            if st.button("Export Match History CSV", key="exp_m"):
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "matches.csv", "text/csv")
