"""
Compatibility engine for blood-type and organ donation matching.
Handles all medical compatibility rules and scoring logic.
"""

import math
from typing import Optional

# ── Blood-type compatibility matrix ───────────────────────────────────────────
# donor -> set of recipients that can accept

BLOOD_DONATE_TO: dict[str, list[str]] = {
    "O-":  ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],  # universal
    "O+":  ["O+", "A+", "B+", "AB+"],
    "A-":  ["A-", "A+", "AB-", "AB+"],
    "A+":  ["A+", "AB+"],
    "B-":  ["B-", "B+", "AB-", "AB+"],
    "B+":  ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}

# recipient -> set of donor blood types that are compatible
BLOOD_RECEIVE_FROM: dict[str, list[str]] = {
    "AB+": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],  # universal
    "AB-": ["O-", "A-", "B-", "AB-"],
    "A+":  ["O-", "O+", "A-", "A+"],
    "A-":  ["O-", "A-"],
    "B+":  ["O-", "O+", "B-", "B+"],
    "B-":  ["O-", "B-"],
    "O+":  ["O-", "O+"],
    "O-":  ["O-"],
}

# ── Organ-specific age constraints ─────────────────────────────────────────────
ORGAN_AGE_LIMITS: dict[str, dict] = {
    "Kidney":          {"donor_max": 70, "donor_min": 18},
    "Liver":           {"donor_max": 65, "donor_min": 18},
    "Heart":           {"donor_max": 55, "donor_min": 18},
    "Lungs":           {"donor_max": 55, "donor_min": 18},
    "Pancreas":        {"donor_max": 50, "donor_min": 18},
    "Cornea":          {"donor_max": 80, "donor_min": 2},
    "Bone Marrow":     {"donor_max": 60, "donor_min": 18},
    "Skin":            {"donor_max": 65, "donor_min": 18},
    "Small Intestine": {"donor_max": 60, "donor_min": 18},
    "Blood":           {"donor_max": 65, "donor_min": 18},
}

# Organs that require strict blood-type matching (not just compatibility)
STRICT_BLOOD_MATCH_ORGANS = {"Heart", "Lungs", "Pancreas", "Small Intestine"}

# Blood donation is always about compatibility, not exact match
BLOOD_COMPAT_ORGANS = {"Blood", "Bone Marrow", "Kidney", "Liver", "Cornea", "Skin"}


def is_blood_compatible(donor_blood: str, recipient_blood: str) -> bool:
    """Return True if donor blood type can donate to recipient."""
    donors_for_recipient = BLOOD_RECEIVE_FROM.get(recipient_blood, [])
    return donor_blood in donors_for_recipient


def blood_compatibility_score(donor_blood: str, recipient_blood: str) -> float:
    """
    Score 0-40 based on blood type compatibility.
      40 = exact match
      30 = compatible (not exact)
       0 = incompatible
    """
    if donor_blood == recipient_blood:
        return 40.0
    if is_blood_compatible(donor_blood, recipient_blood):
        return 30.0
    return 0.0


def organ_compatibility_score(
    organ: str,
    donor_blood: str,
    recipient_blood: str,
    donor_age: int,
) -> float:
    """
    Score 0-40 based on organ + blood type rules.
    Returns 0 if fundamentally incompatible.
    """
    limits = ORGAN_AGE_LIMITS.get(organ, {})
    donor_min = limits.get("donor_min", 18)
    donor_max = limits.get("donor_max", 70)

    if not (donor_min <= donor_age <= donor_max):
        return 0.0   # age hard-fail

    if organ in STRICT_BLOOD_MATCH_ORGANS:
        if donor_blood != recipient_blood:
            return 0.0
        return 40.0

    # For other organs, use blood compatibility
    return blood_compatibility_score(donor_blood, recipient_blood)


def distance_km(
    lat1: Optional[float], lon1: Optional[float],
    lat2: Optional[float], lon2: Optional[float],
) -> Optional[float]:
    """Haversine distance in kilometres. Returns None if coords missing."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def location_score(dist: Optional[float]) -> float:
    """
    Score 0-20 based on proximity.
      20 = same city / < 50 km
      15 = < 200 km
      10 = < 500 km
       5 = < 1000 km
       2 = >= 1000 km  (still possible for organ transport)
    """
    if dist is None:
        return 10.0   # neutral when unknown
    if dist < 50:
        return 20.0
    if dist < 200:
        return 15.0
    if dist < 500:
        return 10.0
    if dist < 1000:
        return 5.0
    return 2.0


def availability_score(is_available: bool) -> float:
    """Score 0 or 10 based on donor availability status."""
    return 10.0 if is_available else 0.0


def urgency_bonus(urgency_level: str) -> float:
    """Extra weight for urgency (used for display/prioritisation, not hard filter)."""
    bonuses = {
        "Critical": 10.0,
        "High": 7.0,
        "Medium": 4.0,
        "Low": 1.0,
    }
    return bonuses.get(urgency_level, 4.0)


def compute_compatibility_score(
    donor_blood: str,
    donor_age: int,
    donor_lat: Optional[float],
    donor_lon: Optional[float],
    donor_is_available: bool,
    donor_donation_types: list[str],
    recipient_blood: str,
    required_organ: str,
    recipient_lat: Optional[float],
    recipient_lon: Optional[float],
    urgency: str = "Medium",
) -> dict:
    """
    Master scoring function. Returns a dict with:
        score       : float 0-100
        breakdown   : dict of sub-scores
        compatible  : bool (hard pass/fail)
        distance_km : float | None
        notes       : list[str]
    """
    notes: list[str] = []

    # 1. Does donor offer this organ?
    if required_organ not in donor_donation_types:
        return {
            "score": 0.0,
            "compatible": False,
            "breakdown": {},
            "distance_km": None,
            "notes": [f"Donor does not offer {required_organ}"],
        }

    # 2. Availability
    avail = availability_score(donor_is_available)
    if avail == 0:
        notes.append("Donor currently unavailable")

    # 3. Blood/organ compatibility
    organ_score = organ_compatibility_score(
        required_organ, donor_blood, recipient_blood, donor_age
    )
    if organ_score == 0.0:
        return {
            "score": 0.0,
            "compatible": False,
            "breakdown": {"organ": 0.0, "location": 0.0, "availability": avail},
            "distance_km": None,
            "notes": notes + [f"Blood type {donor_blood} incompatible with {recipient_blood} for {required_organ}"],
        }

    # 4. Location
    dist = distance_km(donor_lat, donor_lon, recipient_lat, recipient_lon)
    loc_score = location_score(dist)

    if required_organ not in {"Blood", "Cornea", "Bone Marrow", "Skin"}:
        # Solid organs are time-critical; penalise heavily if far
        if dist is not None and dist > 1500:
            notes.append(f"Distance {dist:.0f} km may be problematic for {required_organ} transport")

    # 5. Urgency bonus (normalised to keep total ≤ 100)
    urg = urgency_bonus(urgency)

    # Raw total (max = 40 + 20 + 10 + 10 = 80 before normalisation; scale to 100)
    raw = organ_score + loc_score + avail + urg
    raw_max = 40 + 20 + 10 + 10
    score = round(min((raw / raw_max) * 100, 100), 1)

    if donor_blood == recipient_blood:
        notes.append("Exact blood type match")
    elif organ_score > 0:
        notes.append(f"Blood type {donor_blood} compatible with {recipient_blood}")

    return {
        "score": score,
        "compatible": True,
        "breakdown": {
            "blood_organ": organ_score,
            "location": loc_score,
            "availability": avail,
            "urgency_bonus": urg,
        },
        "distance_km": dist,
        "notes": notes,
    }


def get_compatible_blood_types(recipient_blood: str) -> list[str]:
    """Return all blood types that can donate to the given recipient."""
    return BLOOD_RECEIVE_FROM.get(recipient_blood, [])


def get_donation_type_options() -> list[str]:
    return [
        "Blood", "Kidney", "Liver", "Heart", "Lungs",
        "Pancreas", "Cornea", "Bone Marrow", "Skin", "Small Intestine"
    ]


def get_blood_type_options() -> list[str]:
    return ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def get_urgency_options() -> list[str]:
    return ["Critical", "High", "Medium", "Low"]
