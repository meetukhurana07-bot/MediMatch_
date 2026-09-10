"""
MediMatch — Donor-Recipient Matching Platform
Single-file Streamlit app (Streamlit Cloud compatible).
"""

import math
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from google import genai
from google.genai import types

# ── Load API key from .env file (so you don't re-enter it every time) ──────────
def _load_env_key() -> str:
    """Read GEMINI_API_KEY from .env file if it exists, else from env vars."""
    # Try st.secrets first (Streamlit Cloud)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # Try .env file (local development)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    # Try OS environment variable
    return os.environ.get("GEMINI_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
DATABASE_URL = "sqlite:///donation_platform.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Donor(Base):
    __tablename__ = "donors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    blood_type = Column(String(5), nullable=False)
    email = Column(String(150), nullable=False)
    phone = Column(String(20), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    country = Column(String(100), nullable=False, default="India")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    donation_types = Column(Text, nullable=False)
    medical_notes = Column(Text, nullable=True)
    is_available = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.utcnow)


class UrgentRequest(Base):
    __tablename__ = "urgent_requests"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    blood_type = Column(String(5), nullable=False)
    required_donation = Column(String(50), nullable=False)
    hospital_name = Column(String(200), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    country = Column(String(100), nullable=False, default="India")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    urgency_level = Column(String(20), nullable=False, default="High")
    contact_name = Column(String(100), nullable=False)
    contact_phone = Column(String(20), nullable=False)
    contact_email = Column(String(150), nullable=False)
    medical_description = Column(Text, nullable=True)
    is_fulfilled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    deadline = Column(DateTime, nullable=True)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    donor_id = Column(Integer, nullable=False)
    request_id = Column(Integer, nullable=False)
    compatibility_score = Column(Float, nullable=False)
    distance_km = Column(Float, nullable=True)
    status = Column(String(20), default="Pending")
    ai_analysis = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_session():
    return SessionLocal()


@st.cache_resource
def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if db.query(Donor).count() == 0:
        sample_donors = [
            Donor(name="Arun Kumar", age=28, blood_type="O+", email="arun@example.com",
                  phone="+91-9876543210", city="Mumbai", state="Maharashtra",
                  latitude=19.0760, longitude=72.8777, donation_types="Blood,Kidney", is_available=True,
                  medical_notes="Healthy, non-smoker"),
            Donor(name="Priya Sharma", age=34, blood_type="A+", email="priya@example.com",
                  phone="+91-9876543211", city="Delhi", state="Delhi",
                  latitude=28.6139, longitude=77.2090, donation_types="Blood,Cornea,Bone Marrow", is_available=True),
            Donor(name="Ravi Singh", age=45, blood_type="B+", email="ravi@example.com",
                  phone="+91-9876543212", city="Bangalore", state="Karnataka",
                  latitude=12.9716, longitude=77.5946, donation_types="Blood,Liver", is_available=True),
            Donor(name="Meena Reddy", age=29, blood_type="AB+", email="meena@example.com",
                  phone="+91-9876543213", city="Chennai", state="Tamil Nadu",
                  latitude=13.0827, longitude=80.2707, donation_types="Blood,Kidney,Pancreas", is_available=True),
            Donor(name="Kiran Patel", age=38, blood_type="O-", email="kiran@example.com",
                  phone="+91-9876543214", city="Ahmedabad", state="Gujarat",
                  latitude=23.0225, longitude=72.5714, donation_types="Blood,Skin,Cornea", is_available=True,
                  medical_notes="Universal donor - O-"),
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
# COMPATIBILITY ENGINE
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
    "Kidney": {"donor_max":70,"donor_min":18}, "Liver": {"donor_max":65,"donor_min":18},
    "Heart":  {"donor_max":55,"donor_min":18}, "Lungs": {"donor_max":55,"donor_min":18},
    "Pancreas": {"donor_max":50,"donor_min":18}, "Cornea": {"donor_max":80,"donor_min":2},
    "Bone Marrow": {"donor_max":60,"donor_min":18}, "Skin": {"donor_max":65,"donor_min":18},
    "Small Intestine": {"donor_max":60,"donor_min":18}, "Blood": {"donor_max":65,"donor_min":18},
}
STRICT_BLOOD_ORGANS = {"Heart","Lungs","Pancreas","Small Intestine"}

CITY_COORDS = {
    "Mumbai":(19.0760,72.8777),"Delhi":(28.6139,77.2090),"Bangalore":(12.9716,77.5946),
    "Chennai":(13.0827,80.2707),"Kolkata":(22.5726,88.3639),"Hyderabad":(17.3850,78.4867),
    "Ahmedabad":(23.0225,72.5714),"Pune":(18.5204,73.8567),"Jaipur":(26.9124,75.7873),
    "Lucknow":(26.8467,80.9462),"Surat":(21.1702,72.8311),"Kanpur":(26.4499,80.3319),
    "Nagpur":(21.1458,79.0882),"Patna":(25.5941,85.1376),"Indore":(22.7196,75.8577),
    "Bhopal":(23.2599,77.4126),"Visakhapatnam":(17.6868,83.2185),"Vadodara":(22.3072,73.1812),
    "Coimbatore":(11.0168,76.9558),"Kochi":(9.9312,76.2673),"Other":(20.5937,78.9629),
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
                "notes":["Donor age outside limits for this organ"]}
    if organ in STRICT_BLOOD_ORGANS and donor_blood != rec_blood:
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":[f"Strict blood match required for {organ}"]}
    compatible_donors = BLOOD_RECEIVE_FROM.get(rec_blood, [])
    if donor_blood not in compatible_donors:
        return {"score":0.0,"compatible":False,"breakdown":{},"distance_km":None,
                "notes":[f"{donor_blood} incompatible with {rec_blood}"]}
    organ_score = 40.0 if donor_blood == rec_blood else 30.0
    dist = _haversine(dlat, dlon, rlat, rlon)
    loc_score = (20 if dist is None else 20 if dist<50 else 15 if dist<200 else 10 if dist<500 else 5 if dist<1000 else 2)
    avail_score = 10.0 if available else 0.0
    urg_bonus = {"Critical":10,"High":7,"Medium":4,"Low":1}.get(urgency, 4)
    raw = organ_score + loc_score + avail_score + urg_bonus
    score = round(min(raw / 80 * 100, 100), 1)
    notes = []
    if donor_blood == rec_blood:
        notes.append("Exact blood type match")
    elif organ_score > 0:
        notes.append(f"{donor_blood} compatible with {rec_blood}")
    return {"score":score,"compatible":True,
            "breakdown":{"blood_organ":organ_score,"location":loc_score,
                         "availability":avail_score,"urgency_bonus":urg_bonus},
            "distance_km":dist,"notes":notes}


# ══════════════════════════════════════════════════════════════════════════════
# AI SERVICE
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_FALLBACKS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
SYSTEM_PROMPT = """You are MediMatch AI, an expert medical assistant specializing in blood and organ donation.
Be empathetic, use plain language, cite ABO/Rh rules accurately, and always remind users that AI analysis is advisory only."""

AI_MODEL = "gemini-2.5-flash-preview-05-20"


def _client(api_key):
    return genai.Client(api_key=api_key)


def _cfg(**kw):
    return types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, **kw)


def resolve_model(api_key):
    global AI_MODEL
    c = _client(api_key)
    for name in _MODEL_FALLBACKS:
        try:
            c.models.generate_content(model=name, contents="Hi",
                                       config=types.GenerateContentConfig(max_output_tokens=3))
            AI_MODEL = name
            return name
        except Exception as e:
            if any(x in str(e) for x in ["NOT_FOUND","404","no longer available","deprecated"]):
                continue
            AI_MODEL = name
            return name
    return AI_MODEL


def validate_api_key(api_key):
    if not api_key or len(api_key) < 20:
        return False, "API key too short."
    try:
        model = resolve_model(api_key)
        return True, f"Valid! Using model: {model}"
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid" in err.lower():
            return False, "Invalid API key."
        return False, f"Error: {err}"


def ai_analyse_match(api_key, donor, request, result):
    dist_str = f"{result['distance_km']:.0f} km" if result.get("distance_km") else "unknown"
    bd = result.get("breakdown", {})
    prompt = f"""Analyse this donation match:
DONOR: {donor['name']}, Age {donor['age']}, Blood {donor['blood_type']}, {donor['city']}, Offers: {donor['donation_types']}
PATIENT: {request['patient_name']}, Age {request['age']}, Blood {request['blood_type']}, Needs: {request['required_donation']}
HOSPITAL: {request['hospital_name']}, {request['city']} | Urgency: {request['urgency_level']}
SCORE: {result['score']}/100 | Distance: {dist_str}
Blood/Organ: {bd.get('blood_organ',0)}/40 | Location: {bd.get('location',0)}/20

Provide: 1) Match Summary 2) Medical Considerations 3) Recommended Actions 4) Urgency Assessment"""
    try:
        r = _client(api_key).models.generate_content(
            model=AI_MODEL, contents=prompt,
            config=_cfg(temperature=0.7, max_output_tokens=2048))
        return r.text
    except Exception as e:
        return f"AI unavailable: {e}"


def ai_chat(api_key, user_msg, history, context=None):
    try:
        chat = _client(api_key).chats.create(
            model=AI_MODEL, config=_cfg(temperature=0.7, max_output_tokens=2048),
            history=[types.Content(role=m["role"], parts=[types.Part(text=m["parts"][0])]) for m in history])
        msg = f"[Context: {context}]\n\n{user_msg}" if context else user_msg
        return chat.send_message(msg).text
    except Exception as e:
        return f"AI error: {e}"


def ai_broadcast(api_key, req):
    prompt = f"""Write a concise urgent appeal (max 100 words) for donation sharing:
Patient: {req.get('patient_name')}, Age {req.get('age')} | Needs: {req.get('required_donation')} ({req.get('blood_type')})
Hospital: {req.get('hospital_name')}, {req.get('city')} | Urgency: {req.get('urgency_level')}
Contact: {req.get('contact_name')} — {req.get('contact_phone')}
Make it compassionate with a clear call-to-action."""
    try:
        r = _client(api_key).models.generate_content(
            model=AI_MODEL, contents=prompt, config=_cfg(temperature=0.8, max_output_tokens=200))
        return r.text
    except Exception as e:
        return f"Error: {e}"


def ai_hospital_guide(api_key, city, organ):
    prompt = f"""A patient in {city} needs a {organ} transplant/donation.
Provide: 1) How to find accredited transplant centres near {city} 2) Key organisations (NOTTO, ROTTO) 3) Required documents/steps."""
    try:
        r = _client(api_key).models.generate_content(
            model=AI_MODEL, contents=prompt, config=_cfg(temperature=0.5, max_output_tokens=1024))
        return r.text
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MediMatch — Blood & Organ Donation",
                   page_icon="🩸", layout="wide", initial_sidebar_state="expanded")
init_db()

for k, v in [("gemini_api_key", _load_env_key()),("api_key_validated",False),
             ("chat_history",[]),("active_page","🏠 Home"),("chat_messages",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

# Auto-validate if key was loaded from .env and not yet validated
if st.session_state.gemini_api_key and not st.session_state.api_key_validated:
    ok, _ = validate_api_key(st.session_state.gemini_api_key)
    st.session_state.api_key_validated = ok

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩸 MediMatch")
    st.caption("AI-Powered Donation Platform")
    st.divider()

    selected = st.radio("Navigate", [
        "🏠 Home", "🧑‍⚕️ Register as Donor", "🆘 Post Urgent Request",
        "🔍 Matching Dashboard", "🤖 AI Assistant", "📋 Manage Records"
    ], index=["🏠 Home","🧑‍⚕️ Register as Donor","🆘 Post Urgent Request",
              "🔍 Matching Dashboard","🤖 AI Assistant","📋 Manage Records"
              ].index(st.session_state.active_page))
    st.session_state.active_page = selected

    st.divider()
    st.subheader("🔑 Gemini API Key")
    api_input = st.text_input("API Key", value=st.session_state.gemini_api_key,
                               type="password", placeholder="AIza...",
                               help="Get free key at https://aistudio.google.com/app/apikey")
    if api_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_input
        st.session_state.api_key_validated = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Validate", use_container_width=True):
            if st.session_state.gemini_api_key:
                with st.spinner("Checking..."):
                    ok, msg = validate_api_key(st.session_state.gemini_api_key)
                    st.session_state.api_key_validated = ok
                    (st.success if ok else st.error)(msg)
            else:
                st.warning("Enter a key first")
    with c2:
        st.info("✅ Active" if st.session_state.api_key_validated else "⚪ Not set")

    st.divider()
    db = get_session()
    nd = db.query(Donor).filter(Donor.is_available == True).count()
    nr = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled == False).count()
    nm = db.query(Match).count()
    db.close()
    st.markdown("**📊 Live Stats**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Donors", nd); s2.metric("Requests", nr); s3.metric("Matches", nm)
    st.divider()
    st.caption("⚕️ Advisory tool only. Consult a qualified medical professional.")

page = st.session_state.active_page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🩸 MediMatch — AI-Powered Donation Platform")
    st.markdown("> **Connecting donors and recipients through intelligent matching.**  \n> Register as a donor, post urgent requests, and let our AI engine find the best compatible matches instantly.")

    db = get_session()
    td = db.query(Donor).count()
    ad = db.query(Donor).filter(Donor.is_available==True).count()
    orq = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).count()
    cr = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False, UrgentRequest.urgency_level=="Critical").count()
    tm = db.query(Match).count()
    db.close()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🧑‍⚕️ Total Donors", td); c2.metric("✅ Active Donors", ad)
    c3.metric("📋 Open Requests", orq); c4.metric("🚨 Critical", cr, delta_color="inverse")
    c5.metric("🔗 Matches", tm)
    st.divider()

    db = get_session()
    try:
        donors_df = pd.read_sql("SELECT blood_type, donation_types FROM donors WHERE is_available=1", db.bind)
        req_df = pd.read_sql("SELECT urgency_level, required_donation FROM urgent_requests WHERE is_fulfilled=0", db.bind)
    finally:
        db.close()

    ca, cb, cc = st.columns(3)
    with ca:
        if not donors_df.empty:
            bt = donors_df["blood_type"].value_counts().reset_index()
            bt.columns = ["Blood Type","Count"]
            fig = px.pie(bt, names="Blood Type", values="Count", title="Donors by Blood Type",
                         color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
            fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), height=260)
            st.plotly_chart(fig, use_container_width=True)
    with cb:
        if not req_df.empty:
            urg = req_df["urgency_level"].value_counts().reset_index()
            urg.columns = ["Urgency","Count"]
            fig = px.bar(urg, x="Urgency", y="Count", title="Requests by Urgency",
                         color="Urgency", color_discrete_map={"Critical":"#d62728","High":"#ff7f0e","Medium":"#1f77b4","Low":"#2ca02c"}, text="Count")
            fig.update_layout(showlegend=False, margin=dict(t=40,b=0), height=260)
            st.plotly_chart(fig, use_container_width=True)
    with cc:
        if not donors_df.empty:
            rows = [t.strip() for row in donors_df["donation_types"] for t in str(row).split(",")]
            dt = pd.Series(rows).value_counts().reset_index()
            dt.columns = ["Type","Count"]
            fig = px.bar(dt.head(8), x="Count", y="Type", title="Donors by Type",
                         orientation="h", color="Count", color_continuous_scale="Reds")
            fig.update_layout(margin=dict(t=40,b=0), height=260, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    db = get_session()
    critical_list = db.query(UrgentRequest).filter(
        UrgentRequest.is_fulfilled==False,
        UrgentRequest.urgency_level.in_(["Critical","High"])
    ).order_by(UrgentRequest.created_at.desc()).limit(5).all()
    db.close()
    if critical_list:
        st.subheader("🚨 Active Critical & High-Urgency Requests")
        for req in critical_list:
            badge = "🔴" if req.urgency_level == "Critical" else "🟠"
            with st.expander(f"{badge} {req.urgency_level} — {req.required_donation} | {req.patient_name} | {req.hospital_name}, {req.city}", expanded=req.urgency_level=="Critical"):
                r1,r2,r3 = st.columns(3)
                r1.markdown(f"**Patient:** {req.patient_name}, Age {req.age}\n\n**Blood:** `{req.blood_type}`")
                r2.markdown(f"**Needs:** {req.required_donation}\n\n**Hospital:** {req.hospital_name}")
                r3.markdown(f"**Contact:** {req.contact_name}\n\n**Phone:** {req.contact_phone}")
    else:
        st.success("✅ No critical requests at this time.")
    st.divider()
    st.subheader("Quick Actions")
    q1,q2,q3,q4 = st.columns(4)
    with q1:
        if st.button("🧑‍⚕️ Register as Donor", use_container_width=True, type="primary"):
            st.session_state.active_page = "🧑‍⚕️ Register as Donor"; st.rerun()
    with q2:
        if st.button("🆘 Post Urgent Request", use_container_width=True, type="primary"):
            st.session_state.active_page = "🆘 Post Urgent Request"; st.rerun()
    with q3:
        if st.button("🔍 Find Matches", use_container_width=True):
            st.session_state.active_page = "🔍 Matching Dashboard"; st.rerun()
    with q4:
        if st.button("🤖 Ask AI Assistant", use_container_width=True):
            st.session_state.active_page = "🤖 AI Assistant"; st.rerun()
    st.divider()
    st.markdown("""### How MediMatch Works
| Step | Action | Detail |
|------|--------|--------|
| 1️⃣ | **Register** | Donors sign up with blood type, location & available organs |
| 2️⃣ | **Request** | Hospitals/families post urgent needs with patient details |
| 3️⃣ | **Match** | AI engine scores every donor against each request |
| 4️⃣ | **Analyse** | Gemini 2.5 Flash provides detailed medical analysis |
| 5️⃣ | **Connect** | Matched donors are notified with hospital contact details |""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER DONOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧑‍⚕️ Register as Donor":
    st.title("🧑‍⚕️ Register as a Donor")
    st.markdown("Join our life-saving network. Your registration could save up to **8 lives**.")

    edit_id = st.session_state.get("edit_donor_id")
    prefill = {}
    if edit_id:
        db = get_session()
        d = db.query(Donor).filter(Donor.id == edit_id).first()
        if d:
            prefill = {"name":d.name,"age":d.age,"blood_type":d.blood_type,"email":d.email,
                       "phone":d.phone,"city":d.city,"state":d.state,
                       "donation_types":d.donation_types.split(","),"medical_notes":d.medical_notes or "","is_available":d.is_available}
        db.close()
        st.info(f"✏️ Editing donor ID: {edit_id}")

    with st.form("donor_form", clear_on_submit=False):
        st.subheader("Personal Information")
        f1, f2 = st.columns(2)
        with f1:
            name = st.text_input("Full Name *", value=prefill.get("name",""), placeholder="e.g. Arun Kumar")
            age = st.number_input("Age *", min_value=18, max_value=80, value=int(prefill.get("age",25)))
            blood_type = st.selectbox("Blood Type *", BLOOD_TYPES,
                                       index=BLOOD_TYPES.index(prefill.get("blood_type","O+")))
        with f2:
            email = st.text_input("Email *", value=prefill.get("email",""), placeholder="you@example.com")
            phone = st.text_input("Phone *", value=prefill.get("phone",""), placeholder="+91-XXXXXXXXXX")
            is_available = st.checkbox("Currently available to donate", value=prefill.get("is_available",True))
        st.subheader("Location")
        l1,l2 = st.columns(2)
        with l1:
            city_list = list(CITY_COORDS.keys())
            city = st.selectbox("Nearest City *", city_list,
                                 index=city_list.index(prefill.get("city","Mumbai")) if prefill.get("city") in city_list else 0)
        with l2:
            state = st.selectbox("State *", STATES,
                                  index=STATES.index(prefill.get("state","Maharashtra")) if prefill.get("state") in STATES else 0)
        st.subheader("Donation Preferences")
        pt = prefill.get("donation_types", ["Blood"])
        if isinstance(pt, str): pt = [x.strip() for x in pt.split(",")]
        cols = st.columns(5)
        selected_types = []
        for i, organ in enumerate(ORGAN_TYPES):
            with cols[i % 5]:
                if st.checkbox(organ, value=(organ in pt), key=f"org_{organ}"): selected_types.append(organ)
        medical_notes = st.text_area("Medical Notes (optional)", value=prefill.get("medical_notes",""), height=80)
        st.markdown("---")
        consent = st.checkbox("I confirm all information is accurate and consent to being contacted *")
        submitted = st.form_submit_button("✅ Register / Update Profile", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not name.strip(): errors.append("Full name required.")
        if not email.strip() or "@" not in email: errors.append("Valid email required.")
        if not phone.strip(): errors.append("Phone required.")
        if not selected_types: errors.append("Select at least one donation type.")
        if not consent: errors.append("You must agree to the consent statement.")
        for e in errors: st.error(e)
        if not errors:
            lat, lon = CITY_COORDS.get(city, (20.5937, 78.9629))
            db = get_session()
            try:
                if edit_id:
                    d = db.query(Donor).filter(Donor.id == edit_id).first()
                    if d:
                        d.name=name.strip(); d.age=age; d.blood_type=blood_type; d.email=email.strip()
                        d.phone=phone.strip(); d.city=city; d.state=state; d.latitude=lat; d.longitude=lon
                        d.donation_types=",".join(selected_types); d.medical_notes=medical_notes.strip() or None
                        d.is_available=is_available; db.commit()
                        st.success(f"✅ Updated profile for **{name}**!")
                        st.session_state.edit_donor_id = None
                else:
                    nd = Donor(name=name.strip(), age=age, blood_type=blood_type, email=email.strip(),
                               phone=phone.strip(), city=city, state=state, country="India",
                               latitude=lat, longitude=lon, donation_types=",".join(selected_types),
                               medical_notes=medical_notes.strip() or None, is_available=is_available)
                    db.add(nd); db.commit(); db.refresh(nd)
                    st.success(f"🎉 **Thank you, {name}!** Registered as donor ID: {nd.id}")
                    st.balloons()
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                db.close()

    st.divider()
    st.subheader("📋 Registered Donors")
    db = get_session()
    donors = db.query(Donor).order_by(Donor.registered_at.desc()).all()
    db.close()
    if donors:
        rows = [{"ID":d.id,"Name":d.name,"Age":d.age,"Blood":d.blood_type,"City":d.city,
                 "Offers":d.donation_types,"Available":"✅" if d.is_available else "❌",
                 "Registered":d.registered_at.strftime("%d %b %Y")} for d in donors]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No donors registered yet. Be the first!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POST REQUEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🆘 Post Urgent Request":
    st.title("🆘 Post an Urgent Donation Request")
    st.markdown("Submit an urgent request on behalf of a patient. Our system will immediately scan for compatible donors.")

    with st.expander("📖 Urgency Level Guide"):
        u1,u2,u3,u4 = st.columns(4)
        u1.error("🔴 **Critical** — Life-threatening, hours matter")
        u2.warning("🟠 **High** — Days to act, rapidly deteriorating")
        u3.info("🔵 **Medium** — Weeks available, early action preferred")
        u4.success("🟢 **Low** — Scheduled procedure, ample time")

    with st.form("request_form", clear_on_submit=False):
        st.subheader("Patient Information")
        p1,p2 = st.columns(2)
        with p1:
            patient_name = st.text_input("Patient Name *", placeholder="e.g. Suresh Nair")
            patient_age = st.number_input("Patient Age *", min_value=0, max_value=120, value=35)
            blood_type = st.selectbox("Patient Blood Type *", BLOOD_TYPES)
        with p2:
            required_donation = st.selectbox("Donation Required *", ORGAN_TYPES)
            urgency_level = st.selectbox("Urgency Level *", URGENCY_LEVELS)
            deadline_days = st.number_input("Days until deadline (0=none)", min_value=0, max_value=365, value=0)
        st.subheader("Hospital / Location")
        hospital_name = st.text_input("Hospital Name *", placeholder="e.g. Apollo Hospital")
        h1,h2 = st.columns(2)
        with h1:
            city_list = list(CITY_COORDS.keys())
            req_city = st.selectbox("City *", city_list)
        with h2:
            req_state = st.selectbox("State *", STATES)
        st.subheader("Contact Information")
        ct1,ct2,ct3 = st.columns(3)
        with ct1: contact_name = st.text_input("Contact Person *", placeholder="Doctor / Family member")
        with ct2: contact_phone = st.text_input("Contact Phone *", placeholder="+91-XXXXXXXXXX")
        with ct3: contact_email = st.text_input("Contact Email *", placeholder="doctor@hospital.com")
        medical_description = st.text_area("Medical Description *", height=100,
                                            placeholder="Describe the patient's condition and why the donation is needed...")
        st.markdown("---")
        req_submitted = st.form_submit_button("🚨 Submit Urgent Request", type="primary", use_container_width=True)

    if req_submitted:
        errors = []
        if not patient_name.strip(): errors.append("Patient name required.")
        if not hospital_name.strip(): errors.append("Hospital name required.")
        if not contact_name.strip(): errors.append("Contact person required.")
        if not contact_phone.strip(): errors.append("Contact phone required.")
        if not contact_email.strip() or "@" not in contact_email: errors.append("Valid contact email required.")
        if not medical_description.strip(): errors.append("Medical description required.")
        for e in errors: st.error(e)
        if not errors:
            lat, lon = CITY_COORDS.get(req_city, (20.5937, 78.9629))
            deadline = datetime.utcnow() + timedelta(days=deadline_days) if deadline_days > 0 else None
            db = get_session()
            try:
                new_req = UrgentRequest(
                    patient_name=patient_name.strip(), age=patient_age, blood_type=blood_type,
                    required_donation=required_donation, hospital_name=hospital_name.strip(),
                    city=req_city, state=req_state, country="India", latitude=lat, longitude=lon,
                    urgency_level=urgency_level, contact_name=contact_name.strip(),
                    contact_phone=contact_phone.strip(), contact_email=contact_email.strip(),
                    medical_description=medical_description.strip(), deadline=deadline)
                db.add(new_req); db.commit(); db.refresh(new_req)
                badges = {"Critical":"🔴","High":"🟠","Medium":"🔵","Low":"🟢"}
                st.success(f"{badges.get(urgency_level,'⚪')} Request #{new_req.id} submitted! Go to **Matching Dashboard** to find donors.")
                if st.session_state.get("api_key_validated"):
                    with st.spinner("Generating AI broadcast message..."):
                        alert = ai_broadcast(st.session_state.gemini_api_key, {
                            "patient_name":patient_name,"age":patient_age,"blood_type":blood_type,
                            "required_donation":required_donation,"hospital_name":hospital_name,
                            "city":req_city,"urgency_level":urgency_level,
                            "contact_name":contact_name,"contact_phone":contact_phone})
                    st.subheader("📢 AI Broadcast Message")
                    st.info(alert)
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                db.close()

    st.divider()
    st.subheader("📋 Open Requests")
    db = get_session()
    open_reqs = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).order_by(UrgentRequest.created_at.desc()).all()
    db.close()
    if open_reqs:
        open_reqs.sort(key=lambda r: {"Critical":0,"High":1,"Medium":2,"Low":3}.get(r.urgency_level,99))
        for req in open_reqs:
            badge_map = {"Critical":"🔴","High":"🟠","Medium":"🔵","Low":"🟢"}
            with st.expander(f"{badge_map.get(req.urgency_level,'⚪')} [{req.urgency_level}] {req.required_donation} — {req.patient_name} ({req.blood_type}) | {req.hospital_name}, {req.city}"):
                r1,r2,r3 = st.columns(3)
                r1.markdown(f"**ID:** #{req.id}\n\n**Patient:** {req.patient_name}, Age {req.age}\n\n**Blood:** `{req.blood_type}`")
                r2.markdown(f"**Needs:** {req.required_donation}\n\n**Hospital:** {req.hospital_name}\n\n**Location:** {req.city}, {req.state}")
                r3.markdown(f"**Contact:** {req.contact_name}\n\n**Phone:** {req.contact_phone}\n\n**Email:** {req.contact_email}")
                if req.medical_description: st.caption(req.medical_description)
    else:
        st.success("✅ No open requests at this time.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MATCHING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Matching Dashboard":
    st.title("🔍 Matching Dashboard")
    st.markdown("Run the AI-powered compatibility engine to find the best donors for each urgent request.")

    db = get_session()
    open_requests = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).order_by(UrgentRequest.created_at.desc()).all()
    all_donors = db.query(Donor).filter(Donor.is_available==True).all()
    db.close()

    if not open_requests:
        st.warning("No open requests. Post an urgent request first.")
        if st.button("➕ Post a Request"): st.session_state.active_page = "🆘 Post Urgent Request"; st.rerun()
    elif not all_donors:
        st.warning("No active donors. Register donors first.")
        if st.button("➕ Register a Donor"): st.session_state.active_page = "🧑‍⚕️ Register as Donor"; st.rerun()
    else:
        with st.expander("🔧 Filters", expanded=False):
            fc1,fc2,fc3 = st.columns(3)
            with fc1: filter_urgency = st.multiselect("Urgency", URGENCY_LEVELS, default=URGENCY_LEVELS)
            with fc2: filter_organ = st.multiselect("Donation Type", ORGAN_TYPES, default=[])
            with fc3: min_score = st.slider("Min Score", 0, 100, 30)

        filtered_reqs = [r for r in open_requests
                         if r.urgency_level in filter_urgency
                         and (not filter_organ or r.required_donation in filter_organ)]

        if not filtered_reqs:
            st.info("No requests match filters.")
        else:
            labels = [f"#{r.id} | {r.urgency_level} | {r.required_donation} ({r.blood_type}) — {r.patient_name} @ {r.hospital_name}, {r.city}" for r in filtered_reqs]
            sel_label = st.selectbox("Select Request to Match", labels)
            req = filtered_reqs[labels.index(sel_label)]

            urgency_icon = {"Critical":"🔴","High":"🟠","Medium":"🔵","Low":"🟢"}.get(req.urgency_level,"⚪")
            with st.container(border=True):
                st.markdown(f"### {urgency_icon} Request #{req.id} — {req.urgency_level} Priority")
                rc1,rc2,rc3 = st.columns(3)
                rc1.markdown(f"**Patient:** {req.patient_name}, Age {req.age}\n\n**Blood:** `{req.blood_type}`")
                rc2.markdown(f"**Needs:** {req.required_donation}\n\n**Hospital:** {req.hospital_name}")
                rc3.markdown(f"**Location:** {req.city}, {req.state}\n\n**Contact:** {req.contact_name} | {req.contact_phone}")
                if req.medical_description: st.caption(req.medical_description)

            if st.button("🔍 Run Compatibility Matching", type="primary", use_container_width=True):
                with st.spinner("Scanning donors..."):
                    results = []
                    for donor in all_donors:
                        d_types = [t.strip() for t in donor.donation_types.split(",")]
                        r = compute_score(donor.blood_type, donor.age, donor.latitude, donor.longitude,
                                          donor.is_available, d_types, req.blood_type, req.required_donation,
                                          req.latitude, req.longitude, req.urgency_level)
                        if r["compatible"] and r["score"] >= min_score:
                            results.append({"donor":donor,"result":r})
                    results.sort(key=lambda x: x["result"]["score"], reverse=True)
                    st.session_state[f"matches_{req.id}"] = results
                    db = get_session()
                    for m in results:
                        existing = db.query(Match).filter(Match.donor_id==m["donor"].id, Match.request_id==req.id).first()
                        if existing:
                            existing.compatibility_score = m["result"]["score"]
                        else:
                            db.add(Match(donor_id=m["donor"].id, request_id=req.id,
                                         compatibility_score=m["result"]["score"],
                                         distance_km=m["result"]["distance_km"]))
                    db.commit(); db.close()

            matches = st.session_state.get(f"matches_{req.id}", [])
            if not matches:
                st.info("Click **Run Compatibility Matching** to find donors.")
            else:
                st.success(f"✅ Found **{len(matches)}** compatible donor(s).")
                if len(matches) > 1:
                    names = [m["donor"].name for m in matches[:10]]
                    scores = [m["result"]["score"] for m in matches[:10]]
                    fig = go.Figure(go.Bar(x=scores, y=names, orientation="h",
                        marker_color=["#2ca02c" if s>=70 else "#ff7f0e" if s>=50 else "#1f77b4" for s in scores],
                        text=[f"{s:.0f}" for s in scores], textposition="outside"))
                    fig.update_layout(title="Donor Compatibility Scores", xaxis_title="Score (/100)",
                                       yaxis={"autorange":"reversed"}, height=max(250, 40*len(matches[:10])),
                                       margin=dict(l=120,r=60,t=40,b=30))
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Detailed Match Results")
                for rank, mi in enumerate(matches, 1):
                    donor = mi["donor"]; result = mi["result"]; score = result["score"]
                    colour = "🟢 Excellent" if score>=80 else "🟡 Good" if score>=60 else "🟠 Fair" if score>=40 else "🔴 Low"
                    with st.expander(f"Rank #{rank} | {colour} ({score}/100) — {donor.name} | {donor.blood_type} | {donor.city}", expanded=(rank==1)):
                        mc1,mc2,mc3 = st.columns([2,2,1])
                        with mc1:
                            st.markdown(f"**Donor:** {donor.name}, Age {donor.age}\n\n**Blood:** `{donor.blood_type}`\n\n**Location:** {donor.city}, {donor.state}\n\n**Offers:** {donor.donation_types}\n\n**Phone:** {donor.phone}\n\n**Email:** {donor.email}")
                        with mc2:
                            bd = result.get("breakdown",{})
                            st.markdown("**Score Breakdown**")
                            st.progress(int(bd.get("blood_organ",0)/40*100), text=f"Blood/Organ: {bd.get('blood_organ',0):.0f}/40")
                            st.progress(int(bd.get("location",0)/20*100), text=f"Location: {bd.get('location',0):.0f}/20")
                            st.progress(int(bd.get("availability",0)/10*100), text=f"Availability: {bd.get('availability',0):.0f}/10")
                            st.progress(int(bd.get("urgency_bonus",0)/10*100), text=f"Urgency Bonus: {bd.get('urgency_bonus',0):.0f}/10")
                            if result.get("distance_km"): st.metric("📍 Distance", f"{result['distance_km']:.0f} km")
                            for note in result.get("notes",[]): st.info(f"ℹ️ {note}")
                        with mc3:
                            gauge = go.Figure(go.Indicator(mode="gauge+number", value=score,
                                title={"text":"Score","font":{"size":13}},
                                gauge={"axis":{"range":[0,100]},"bar":{"color":"#1f77b4"},
                                       "steps":[{"range":[0,40],"color":"#ffcccc"},{"range":[40,70],"color":"#ffe5cc"},{"range":[70,100],"color":"#ccffcc"}]},
                                number={"suffix":"/100"}))
                            gauge.update_layout(height=190, margin=dict(t=30,b=0,l=10,r=10))
                            st.plotly_chart(gauge, use_container_width=True, key=f"g_{req.id}_{donor.id}")

                        ai_key = f"ai_{req.id}_{donor.id}"
                        if st.session_state.get("api_key_validated"):
                            if st.button("🤖 Generate AI Analysis", key=f"ai_btn_{req.id}_{donor.id}", use_container_width=True):
                                with st.spinner("Gemini 2.5 Flash analysing..."):
                                    analysis = ai_analyse_match(
                                        st.session_state.gemini_api_key,
                                        {"name":donor.name,"age":donor.age,"blood_type":donor.blood_type,
                                         "city":donor.city,"state":donor.state,"donation_types":donor.donation_types,
                                         "medical_notes":donor.medical_notes,"is_available":donor.is_available},
                                        {"patient_name":req.patient_name,"age":req.age,"blood_type":req.blood_type,
                                         "required_donation":req.required_donation,"hospital_name":req.hospital_name,
                                         "city":req.city,"state":req.state,"urgency_level":req.urgency_level,
                                         "medical_description":req.medical_description},
                                        result)
                                st.session_state[ai_key] = analysis
                                db = get_session()
                                ex = db.query(Match).filter(Match.donor_id==donor.id, Match.request_id==req.id).first()
                                if ex: ex.ai_analysis = analysis; db.commit()
                                db.close()
                        else:
                            st.caption("🔑 Add & validate your Gemini API key in the sidebar for AI analysis.")
                        if st.session_state.get(ai_key):
                            st.markdown("---\n#### 🤖 AI Medical Analysis")
                            st.markdown(st.session_state[ai_key])

                st.divider()
                if st.button("✅ Mark Request as Fulfilled", use_container_width=True):
                    db = get_session()
                    r = db.query(UrgentRequest).filter(UrgentRequest.id==req.id).first()
                    if r: r.is_fulfilled = True; db.commit()
                    db.close()
                    st.success(f"Request #{req.id} marked as fulfilled! 🎉")
                    st.session_state.pop(f"matches_{req.id}", None)
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Assistant":
    st.title("🤖 AI Medical Assistant")
    st.markdown("Chat with **MediMatch AI** (Gemini 2.5 Flash) for guidance on blood type compatibility, organ donation rules, urgency assessment, and more.")

    if not st.session_state.get("api_key_validated"):
        st.warning("⚠️ Please enter and validate your **Gemini API Key** in the sidebar.")
    else:
        cfg1, cfg2 = st.columns([3,1])
        with cfg1:
            include_ctx = st.toggle("Include live platform data in AI context", value=True)
        with cfg2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []; st.rerun()

        st.markdown("**Quick Questions:**")
        qcols = st.columns(4)
        quick_prompts = ["What blood types are compatible with O+?","Can a 65-year-old donate a kidney?",
                         "What is the max transport time for a donor heart?","How does bone marrow matching work?"]
        quick_sel = None
        for i, qp in enumerate(quick_prompts):
            with qcols[i]:
                if st.button(qp, key=f"qp_{i}", use_container_width=True): quick_sel = qp
        st.divider()

        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_messages:
                st.info("👋 Hello! I'm MediMatch AI. Ask me anything about blood types, organ compatibility, or donation processes.")
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "🧑"):
                    st.markdown(msg["content"])

        user_input = st.chat_input("Ask MediMatch AI a question...")
        if quick_sel: user_input = quick_sel
        if user_input:
            st.session_state.chat_messages.append({"role":"user","content":user_input})
            with chat_container:
                with st.chat_message("user", avatar="🧑"): st.markdown(user_input)
            history = [{"role":"user" if m["role"]=="user" else "model","parts":[m["content"]]}
                       for m in st.session_state.chat_messages[:-1]]
            ctx = None
            if include_ctx:
                db = get_session()
                nd = db.query(Donor).filter(Donor.is_available==True).count()
                nr = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False).count()
                nc = db.query(UrgentRequest).filter(UrgentRequest.is_fulfilled==False, UrgentRequest.urgency_level=="Critical").count()
                db.close()
                ctx = f"{nd} active donors, {nr} open requests ({nc} critical)."
            with st.spinner("MediMatch AI is thinking..."):
                response = ai_chat(st.session_state.gemini_api_key, user_input, history, ctx)
            st.session_state.chat_messages.append({"role":"assistant","content":response})
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"): st.markdown(response)

        st.divider()
        with st.expander("📖 Blood Type Compatibility Reference"):
            st.markdown("**Can Donate To (Donor → Recipients)**")
            st.table({k:", ".join(v) for k,v in BLOOD_DONATE_TO.items()})
            st.markdown("**Can Receive From (Recipient ← Donors)**")
            st.table({k:", ".join(v) for k,v in BLOOD_RECEIVE_FROM.items()})

        with st.expander("🏥 Find Transplant Centre Guidance"):
            gc1, gc2 = st.columns(2)
            with gc1: guide_city = st.text_input("City", placeholder="e.g. Mumbai")
            with gc2: guide_organ = st.selectbox("Organ", ORGAN_TYPES, key="guide_organ")
            if st.button("🔍 Get Guidance", key="hosp_guide"):
                if guide_city:
                    with st.spinner("Asking Gemini 2.5 Flash..."):
                        st.markdown(ai_hospital_guide(st.session_state.gemini_api_key, guide_city, guide_organ))
                else:
                    st.warning("Enter a city name.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MANAGE RECORDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Manage Records":
    st.title("📋 Manage Records")
    st.markdown("View, edit, and manage all donors, requests, and match history.")

    tab_d, tab_r, tab_m = st.tabs(["🧑‍⚕️ Donors", "🆘 Requests", "🔗 Match History"])

    with tab_d:
        st.subheader("All Registered Donors")
        db = get_session()
        donors = db.query(Donor).order_by(Donor.registered_at.desc()).all()
        db.close()
        if not donors:
            st.info("No donors registered yet.")
        else:
            search = st.text_input("🔍 Search (name, city, blood type)", key="dsearch")
            bt_f = st.multiselect("Blood Type Filter", BLOOD_TYPES, key="dbt")
            av_f = st.radio("Availability", ["All","Available only","Unavailable only"], horizontal=True, key="dav")
            filt = donors
            if search: s=search.lower(); filt=[d for d in filt if s in d.name.lower() or s in d.city.lower() or s in d.blood_type.lower()]
            if bt_f: filt=[d for d in filt if d.blood_type in bt_f]
            if av_f=="Available only": filt=[d for d in filt if d.is_available]
            elif av_f=="Unavailable only": filt=[d for d in filt if not d.is_available]
            st.markdown(f"Showing **{len(filt)}** / {len(donors)} donors")
            for donor in filt:
                ab = "✅" if donor.is_available else "❌"
                with st.expander(f"{ab} {donor.name} | `{donor.blood_type}` | {donor.city} | ID:{donor.id}"):
                    dc1,dc2,dc3 = st.columns(3)
                    dc1.markdown(f"**Name:** {donor.name}\n\n**Age:** {donor.age}\n\n**Blood:** `{donor.blood_type}`")
                    dc2.markdown(f"**Email:** {donor.email}\n\n**Phone:** {donor.phone}\n\n**Location:** {donor.city}, {donor.state}")
                    dc3.markdown(f"**Types:** {donor.donation_types}\n\n**Available:** {ab}\n\n**Registered:** {donor.registered_at.strftime('%d %b %Y')}")
                    if donor.medical_notes: st.caption(f"Notes: {donor.medical_notes}")
                    b1,b2,b3 = st.columns(3)
                    with b1:
                        if st.button("✏️ Edit", key=f"ed_{donor.id}", use_container_width=True):
                            st.session_state.edit_donor_id = donor.id
                            st.session_state.active_page = "🧑‍⚕️ Register as Donor"; st.rerun()
                    with b2:
                        na = not donor.is_available
                        if st.button("✅ Set Available" if na else "⏸️ Set Unavailable", key=f"av_{donor.id}", use_container_width=True):
                            db = get_session(); d = db.query(Donor).filter(Donor.id==donor.id).first()
                            if d: d.is_available=na; db.commit()
                            db.close(); st.rerun()
                    with b3:
                        if st.button("🗑️ Delete", key=f"dd_{donor.id}", use_container_width=True):
                            db = get_session(); d = db.query(Donor).filter(Donor.id==donor.id).first()
                            if d: db.delete(d); db.commit()
                            db.close(); st.success(f"{donor.name} deleted."); st.rerun()
            if st.button("📥 Export Donors CSV", key="exp_d"):
                df = pd.DataFrame([{"ID":d.id,"Name":d.name,"Age":d.age,"Blood":d.blood_type,"City":d.city,"State":d.state,"Types":d.donation_types,"Available":d.is_available} for d in filt])
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), "donors.csv","text/csv")

    with tab_r:
        st.subheader("All Donation Requests")
        db = get_session()
        all_reqs = db.query(UrgentRequest).order_by(UrgentRequest.created_at.desc()).all()
        db.close()
        if not all_reqs:
            st.info("No requests posted yet.")
        else:
            show_f = st.toggle("Show fulfilled requests", value=False, key="show_ful")
            disp = [r for r in all_reqs if show_f or not r.is_fulfilled]
            for req in disp:
                icon = {"Critical":"🔴","High":"🟠","Medium":"🔵","Low":"🟢"}.get(req.urgency_level,"⚪")
                status = "✅ Fulfilled" if req.is_fulfilled else "⏳ Open"
                with st.expander(f"{icon} [{req.urgency_level}] {req.required_donation} — {req.patient_name} | {status} | ID:{req.id}"):
                    rc1,rc2,rc3 = st.columns(3)
                    rc1.markdown(f"**Patient:** {req.patient_name}, Age {req.age}\n\n**Blood:** `{req.blood_type}`\n\n**Needs:** {req.required_donation}")
                    rc2.markdown(f"**Hospital:** {req.hospital_name}\n\n**Location:** {req.city}, {req.state}\n\n**Urgency:** {req.urgency_level}")
                    rc3.markdown(f"**Contact:** {req.contact_name}\n\n**Phone:** {req.contact_phone}\n\n**Posted:** {req.created_at.strftime('%d %b %Y')}")
                    if req.medical_description: st.caption(req.medical_description)
                    rb1,rb2,rb3 = st.columns(3)
                    with rb1:
                        if not req.is_fulfilled and st.button("✅ Mark Fulfilled", key=f"ful_{req.id}", use_container_width=True):
                            db = get_session(); r2 = db.query(UrgentRequest).filter(UrgentRequest.id==req.id).first()
                            if r2: r2.is_fulfilled=True; db.commit()
                            db.close(); st.rerun()
                    with rb2:
                        if st.button("🔍 Match Donors", key=f"mr_{req.id}", use_container_width=True):
                            st.session_state.active_page="🔍 Matching Dashboard"; st.rerun()
                    with rb3:
                        if st.button("🗑️ Delete", key=f"dr_{req.id}", use_container_width=True):
                            db = get_session(); r2 = db.query(UrgentRequest).filter(UrgentRequest.id==req.id).first()
                            if r2: db.delete(r2); db.commit()
                            db.close(); st.success(f"Request #{req.id} deleted."); st.rerun()

    with tab_m:
        st.subheader("Match History")
        db = get_session()
        matches = db.query(Match).order_by(Match.created_at.desc()).all()
        donor_map = {d.id:d for d in db.query(Donor).all()}
        req_map = {r.id:r for r in db.query(UrgentRequest).all()}
        db.close()
        if not matches:
            st.info("No matches yet. Run the Matching Dashboard first.")
        else:
            rows = []
            for m in matches:
                d = donor_map.get(m.donor_id); r = req_map.get(m.request_id)
                rows.append({"ID":m.id,"Donor":d.name if d else f"ID{m.donor_id}","Blood":d.blood_type if d else "?",
                             "Patient":r.patient_name if r else f"ID{m.request_id}","Needs":r.required_donation if r else "?",
                             "Score":f"{m.compatibility_score:.1f}/100","Distance":f"{m.distance_km:.0f}km" if m.distance_km else "?",
                             "Status":m.status,"Date":m.created_at.strftime("%d %b %Y"),"AI":"✅" if m.ai_analysis else "—"})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            ai_matches = [m for m in matches if m.ai_analysis]
            if ai_matches:
                sel_id = st.selectbox("View AI Analysis for Match ID", [m.id for m in ai_matches], key="vm")
                sel = next((m for m in ai_matches if m.id==sel_id), None)
                if sel:
                    d = donor_map.get(sel.donor_id); r = req_map.get(sel.request_id)
                    st.markdown(f"**{d.name if d else '?'} → {r.patient_name if r else '?'}** (Score: {sel.compatibility_score:.1f}/100)")
                    st.markdown(sel.ai_analysis)
            if st.button("📥 Export Match CSV", key="exp_m"):
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), "matches.csv","text/csv")
