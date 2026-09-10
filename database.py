"""
Database models and session management for the Donor-Recipient Matching Platform.
Uses SQLite via SQLAlchemy ORM.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, Text, Enum
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import enum

DATABASE_URL = "sqlite:///donation_platform.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Enums ──────────────────────────────────────────────────────────────────────

class DonationType(str, enum.Enum):
    BLOOD = "Blood"
    KIDNEY = "Kidney"
    LIVER = "Liver"
    HEART = "Heart"
    LUNGS = "Lungs"
    PANCREAS = "Pancreas"
    CORNEA = "Cornea"
    BONE_MARROW = "Bone Marrow"
    SKIN = "Skin"
    SMALL_INTESTINE = "Small Intestine"


class UrgencyLevel(str, enum.Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class BloodType(str, enum.Enum):
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"


class MatchStatus(str, enum.Enum):
    PENDING = "Pending"
    NOTIFIED = "Notified"
    ACCEPTED = "Accepted"
    DECLINED = "Declined"
    COMPLETED = "Completed"


# ── Models ─────────────────────────────────────────────────────────────────────

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
    donation_types = Column(Text, nullable=False)       # comma-separated list
    medical_notes = Column(Text, nullable=True)
    is_available = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    compatibility_score = Column(Float, nullable=False)   # 0-100
    distance_km = Column(Float, nullable=True)
    status = Column(String(20), default=MatchStatus.PENDING)
    ai_analysis = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIConversation(Base):
    __tablename__ = "ai_conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── DB helpers ─────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables and seed sample data on first run."""
    Base.metadata.create_all(bind=engine)
    _seed_sample_data()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session():
    """Return a plain (non-generator) session for direct use."""
    return SessionLocal()


def _seed_sample_data():
    """Insert demo donors and requests only when the DB is empty."""
    db = SessionLocal()
    if db.query(Donor).count() > 0:
        db.close()
        return

    sample_donors = [
        Donor(name="Arun Kumar", age=28, blood_type="O+", email="arun@example.com",
              phone="+91-9876543210", city="Mumbai", state="Maharashtra", country="India",
              latitude=19.0760, longitude=72.8777,
              donation_types="Blood,Kidney", is_available=True,
              medical_notes="Healthy, non-smoker"),
        Donor(name="Priya Sharma", age=34, blood_type="A+", email="priya@example.com",
              phone="+91-9876543211", city="Delhi", state="Delhi", country="India",
              latitude=28.6139, longitude=77.2090,
              donation_types="Blood,Cornea,Bone Marrow", is_available=True,
              medical_notes="No prior medical issues"),
        Donor(name="Ravi Singh", age=45, blood_type="B+", email="ravi@example.com",
              phone="+91-9876543212", city="Bangalore", state="Karnataka", country="India",
              latitude=12.9716, longitude=77.5946,
              donation_types="Blood,Liver", is_available=True),
        Donor(name="Meena Reddy", age=29, blood_type="AB+", email="meena@example.com",
              phone="+91-9876543213", city="Chennai", state="Tamil Nadu", country="India",
              latitude=13.0827, longitude=80.2707,
              donation_types="Blood,Kidney,Pancreas", is_available=True),
        Donor(name="Kiran Patel", age=38, blood_type="O-", email="kiran@example.com",
              phone="+91-9876543214", city="Ahmedabad", state="Gujarat", country="India",
              latitude=23.0225, longitude=72.5714,
              donation_types="Blood,Skin,Cornea", is_available=True,
              medical_notes="Universal donor - O-"),
    ]

    sample_requests = [
        UrgentRequest(patient_name="Suresh Nair", age=52, blood_type="O+",
                      required_donation="Kidney", hospital_name="Apollo Hospital",
                      city="Mumbai", state="Maharashtra", country="India",
                      latitude=19.1136, longitude=72.8697,
                      urgency_level="Critical", contact_name="Dr. Sharma",
                      contact_phone="+91-9000001111", contact_email="apollo@example.com",
                      medical_description="End-stage renal disease, dialysis 3x/week"),
        UrgentRequest(patient_name="Lalitha Devi", age=41, blood_type="A+",
                      required_donation="Blood", hospital_name="AIIMS Delhi",
                      city="Delhi", state="Delhi", country="India",
                      latitude=28.5672, longitude=77.2100,
                      urgency_level="High", contact_name="Dr. Verma",
                      contact_phone="+91-9000002222", contact_email="aiims@example.com",
                      medical_description="Post-surgery anemia, urgent transfusion needed"),
    ]

    db.add_all(sample_donors)
    db.add_all(sample_requests)
    db.commit()
    db.close()
