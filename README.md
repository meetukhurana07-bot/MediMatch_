# 🩸 MediMatch — AI-Powered Blood & Organ Donation Platform

An intelligent, Streamlit-based donor-recipient matching platform powered by **Gemini 2.5 Flash**.

## Features

| Feature | Description |
|---------|-------------|
| 🧑‍⚕️ Donor Registration | Register with blood type, location, and organ preferences |
| 🆘 Urgent Requests | Hospitals post critical needs with urgency levels |
| 🔍 Smart Matching | Automatic scoring based on blood type, organ rules, distance & availability |
| 🤖 AI Analysis | Gemini 2.5 Flash narrates detailed match assessments |
| 💬 AI Chat | Multi-turn medical assistant for donation guidance |
| 📋 Record Management | View, edit, delete donors/requests, export CSV |
| 📊 Live Dashboard | Real-time charts: donors by blood type, requests by urgency, organ distribution |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Add your Gemini API Key
- Open the sidebar → paste your key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Click **Validate** — AI features activate instantly

## Project Structure

```
├── app.py                    # Main Streamlit entry point
├── database.py               # SQLAlchemy models + SQLite init
├── compatibility.py          # Blood type + organ matching engine
├── ai_service.py             # Gemini 2.5 Flash integration
├── requirements.txt
├── .streamlit/config.toml    # Theme config
└── pages/
    ├── home.py               # Landing page with live stats
    ├── register_donor.py     # Donor registration form
    ├── post_request.py       # Urgent request form
    ├── matching_dashboard.py # Run matching + AI analysis
    ├── ai_assistant.py       # Multi-turn AI chat
    └── manage_records.py     # CRUD + match history
```

## Compatibility Logic

**Blood Type Scoring (0–40 pts)**
- Exact match: 40 pts
- Compatible (not exact): 30 pts
- Incompatible: 0 pts (hard reject)

**Organ Rules**
- Heart, Lungs, Pancreas, Small Intestine: strict exact blood-type match required
- All organs: donor age constraints enforced

**Location Scoring (0–20 pts)**
- < 50 km: 20 pts | < 200 km: 15 pts | < 500 km: 10 pts | < 1000 km: 5 pts

**Urgency Bonus (0–10 pts)**
- Critical: +10 | High: +7 | Medium: +4 | Low: +1

---
> ⚕️ This platform is an advisory tool. Always consult a qualified medical professional for donation decisions.
