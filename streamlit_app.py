"""
streamlit_app.py
-----------------
PulsePredict - Patient Health Outcome Predictor (single-file Streamlit app)

Everything the previous 3-file version split across clinical_data.py /
predictor.py / app.py now lives in this one file, plus a new top navigation
bar with 6 sections:

    Dashboard | New Prediction | Patient History |
    Disease Analysis | Treatment Analysis | Risk Analysis

Persistence: every saved assessment is POSTed to a Flask backend
(flask_backend.py) which stores it in database.db (SQLite). Patient
History / Dashboard / the 3 analysis tabs all read from that database, so
your data survives across Streamlit restarts -- not just session memory.

Run (two terminals):
    Terminal 1:  python flask_backend.py        # starts DB + API on :5000
    Terminal 2:  streamlit run streamlit_app.py  # starts the UI
"""

import copy
import json
import math
import random
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="PulsePredict - Patient Health Outcome Predictor",
    page_icon="💓",
    layout="wide",
)

API_BASE_URL_DEFAULT = "http://127.0.0.1:5000"


# ===========================================================================
# SECTION A: Clinical reference data  (formerly clinical_data.py)
# ===========================================================================
DEFAULT_PATIENT = {
    "Age": 45, "Gender": "Male", "Living_Area": "Urban",
    "Systolic_BP": 120, "Diastolic_BP": 80,
    "Glucose": 100, "Cholesterol": 180, "BMI": 24.5,
    "Diabetes": 0, "Hypertension": 0, "Heart_Disease": 0,
    "Past_Diseases": "None",
    "Physical_Activity_Hours_Week": 2.5, "Sleep_Hours_Per_Night": 7.0,
    "Diet_Quality": "Average", "Smoking_Status": "Never", "Stress_Level": "Moderate",
    "Treatment": "Medication", "Treatment_Duration": 30, "Follow_Up": "Yes",
}

LEVEL_EMOJI = {"good": "🟢", "watch": "🟡", "warn": "🟠", "bad": "🔴", "low": "🔵", "critical": "🟣"}


def getBPCategory(systolic, diastolic):
    systolic, diastolic = systolic or 0, diastolic or 0
    if systolic < 90 or diastolic < 60:
        return {"label": "Hypotension", "level": "low"}
    if systolic < 120 and diastolic < 80:
        return {"label": "Normal", "level": "good"}
    if systolic < 130 and diastolic < 80:
        return {"label": "Elevated", "level": "watch"}
    if systolic < 140 or diastolic < 90:
        return {"label": "Hypertension Stage 1", "level": "warn"}
    if systolic < 180 and diastolic < 120:
        return {"label": "Hypertension Stage 2", "level": "bad"}
    return {"label": "Hypertensive Crisis", "level": "critical"}


def getGlucoseCategory(glucose):
    glucose = glucose or 0
    if glucose < 70:
        return {"label": "Hypoglycemia", "level": "warn"}
    if glucose <= 99:
        return {"label": "Normal", "level": "good"}
    if glucose <= 125:
        return {"label": "Prediabetic Range", "level": "watch"}
    return {"label": "Diabetic Range", "level": "bad"}


def getCholesterolCategory(chol):
    chol = chol or 0
    if chol < 200:
        return {"label": "Desirable", "level": "good"}
    if chol < 240:
        return {"label": "Borderline High", "level": "watch"}
    return {"label": "High", "level": "bad"}


def getBMICategory(bmi):
    bmi = bmi or 0
    if bmi < 18.5:
        return {"label": "Underweight", "level": "watch"}
    if bmi < 25:
        return {"label": "Normal", "level": "good"}
    if bmi < 30:
        return {"label": "Overweight", "level": "watch"}
    return {"label": "Obese", "level": "bad"}


PATIENT_PERSONAS = [
    {"id": "optimal", "name": "Optimal Health Profile", "tag": "Low Risk",
     "description": "Healthy vitals, active lifestyle, full adherence.",
     "patient": {**DEFAULT_PATIENT, "Age": 34, "Systolic_BP": 115, "Diastolic_BP": 75,
                 "Glucose": 88, "Cholesterol": 170, "BMI": 22.0, "Physical_Activity_Hours_Week": 6.0,
                 "Sleep_Hours_Per_Night": 8.0, "Diet_Quality": "Good", "Stress_Level": "Low",
                 "Treatment": "Lifestyle Changes", "Treatment_Duration": 20}},
    {"id": "high-risk", "name": "High-Risk Comorbid Patient", "tag": "High Risk",
     "description": "Multiple chronic conditions, poor lifestyle indicators.",
     "patient": {**DEFAULT_PATIENT, "Age": 67, "Systolic_BP": 162, "Diastolic_BP": 98,
                 "Glucose": 185, "Cholesterol": 265, "BMI": 33.5, "Diabetes": 1, "Hypertension": 1,
                 "Heart_Disease": 1, "Past_Diseases": "Kidney Disease", "Physical_Activity_Hours_Week": 0.5,
                 "Sleep_Hours_Per_Night": 5.0, "Diet_Quality": "Poor", "Smoking_Status": "Current",
                 "Stress_Level": "High", "Treatment_Duration": 180, "Follow_Up": "No"}},
    {"id": "post-surgical", "name": "Post-Surgical Recovery", "tag": "Monitoring",
     "description": "Recent surgery, moderate vitals, good compliance.",
     "patient": {**DEFAULT_PATIENT, "Age": 52, "Systolic_BP": 128, "Diastolic_BP": 84,
                 "Glucose": 110, "Cholesterol": 205, "BMI": 26.0, "Hypertension": 1,
                 "Past_Diseases": "Arthritis", "Physical_Activity_Hours_Week": 1.5,
                 "Sleep_Hours_Per_Night": 6.5, "Smoking_Status": "Former",
                 "Treatment": "Surgery", "Treatment_Duration": 14}},
    {"id": "young-lifestyle", "name": "Young Adult, Lifestyle Risk", "tag": "Lifestyle Watch",
     "description": "Young patient with modifiable lifestyle risk factors.",
     "patient": {**DEFAULT_PATIENT, "Age": 26, "Systolic_BP": 124, "Diastolic_BP": 79,
                 "Glucose": 98, "Cholesterol": 195, "BMI": 27.5, "Physical_Activity_Hours_Week": 1.0,
                 "Sleep_Hours_Per_Night": 5.5, "Diet_Quality": "Poor", "Smoking_Status": "Current",
                 "Stress_Level": "High", "Treatment": "Lifestyle Changes", "Treatment_Duration": 10}},
]


def generateClinicalRecommendations(patient, probability):
    recs = []
    bp = getBPCategory(patient.get("Systolic_BP"), patient.get("Diastolic_BP"))
    if bp["level"] in ("warn", "bad", "critical"):
        recs.append({"title": "Blood Pressure Management",
                      "detail": f"Classified as '{bp['label']}'. Recommend antihypertensive review and home BP monitoring.",
                      "priority": "high" if bp["level"] in ("bad", "critical") else "medium"})
    gl = getGlucoseCategory(patient.get("Glucose"))
    if gl["level"] in ("watch", "bad"):
        recs.append({"title": "Glycemic Control",
                      "detail": f"Glucose in '{gl['label']}' range. Suggest HbA1c testing and dietary review.",
                      "priority": "high" if gl["level"] == "bad" else "medium"})
    chol = getCholesterolCategory(patient.get("Cholesterol"))
    if chol["level"] in ("watch", "bad"):
        recs.append({"title": "Lipid Profile Follow-up",
                      "detail": f"Cholesterol is '{chol['label']}'. Consider statin evaluation and fasting lipid panel.",
                      "priority": "medium" if chol["level"] == "watch" else "high"})
    bmi = getBMICategory(patient.get("BMI"))
    if bmi["level"] in ("watch", "bad"):
        recs.append({"title": "Weight Management",
                      "detail": f"BMI is '{bmi['label']}'. Recommend nutrition counseling and activity ramp-up.",
                      "priority": "medium"})
    if patient.get("Smoking_Status") == "Current":
        recs.append({"title": "Smoking Cessation",
                      "detail": "Active smoking impairs recovery. Offer cessation counseling / NRT.",
                      "priority": "high"})
    if (patient.get("Physical_Activity_Hours_Week") or 0) < 2.5:
        recs.append({"title": "Increase Physical Activity",
                      "detail": "Below WHO-recommended 2.5 hrs/week. Suggest graded activity program.",
                      "priority": "medium"})
    if (patient.get("Sleep_Hours_Per_Night") or 0) < 6:
        recs.append({"title": "Sleep Hygiene",
                      "detail": "Sleep below 7-9h range may delay recovery. Consider sleep hygiene consult.",
                      "priority": "low"})
    if patient.get("Stress_Level") == "High":
        recs.append({"title": "Stress & Mental Health Support",
                      "detail": "High stress linked to poorer adherence. Recommend counseling referral.",
                      "priority": "medium"})
    if patient.get("Follow_Up") != "Yes":
        recs.append({"title": "Follow-Up Appointment Adherence",
                      "detail": "No follow-up attended -- strongest positive predictor. Reschedule immediately.",
                      "priority": "high"})
    if patient.get("Diabetes") or patient.get("Hypertension") or patient.get("Heart_Disease"):
        recs.append({"title": "Chronic Condition Co-Management",
                      "detail": "Comorbidities present. Recommend coordinated specialist care.",
                      "priority": "medium"})
    if not recs:
        recs.append({"title": "Maintain Current Regimen",
                      "detail": "All major biomarkers within healthy ranges. Continue routine check-ups.",
                      "priority": "low"})
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order.get(r["priority"], 3))
    return recs


# ===========================================================================
# SECTION B: Offline prediction engine  (formerly predictor.py)
# ===========================================================================
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict_offline(patient):
    """Transparent weighted-scoring stand-in for a trained ML model -- swap
    this out for model.predict_proba(...) once you have a real trained
    model.pkl; the rest of the app only cares about the returned dict shape."""
    factors = []
    logit = 0.0

    def add(name, description, weight):
        nonlocal logit
        logit += weight
        factors.append({"name": name, "description": description,
                         "impact": round(weight, 3), "magnitude": abs(weight),
                         "isPositive": weight > 0})

    add("Follow-Up Adherence",
        "Attended scheduled follow-up" if patient.get("Follow_Up") == "Yes" else "Missed follow-up appointment",
        0.275 if patient.get("Follow_Up") == "Yes" else -0.275)

    if patient.get("Diabetes"):
        add("Diabetes", "Diagnosed diabetes mellitus", -0.22)
    if patient.get("Hypertension"):
        add("Hypertension", "Diagnosed hypertension", -0.15)
    if patient.get("Heart_Disease"):
        add("Heart Disease", "Diagnosed cardiac history", -0.30)

    bmi = patient.get("BMI") or 24
    if bmi < 18.5 or bmi >= 30:
        add("BMI", f"BMI of {bmi} outside healthy range", -0.15)
    elif bmi >= 25:
        add("BMI", f"BMI of {bmi} slightly elevated", -0.07)
    else:
        add("BMI", f"BMI of {bmi} within healthy range", 0.08)

    glucose = patient.get("Glucose") or 100
    if glucose >= 126:
        add("Blood Glucose", f"{glucose} mg/dL - diabetic range", -0.18)
    elif glucose >= 100:
        add("Blood Glucose", f"{glucose} mg/dL - prediabetic range", -0.08)
    else:
        add("Blood Glucose", f"{glucose} mg/dL - normal range", 0.07)

    chol = patient.get("Cholesterol") or 180
    if chol >= 240:
        add("Cholesterol", f"{chol} mg/dL - high", -0.14)
    elif chol >= 200:
        add("Cholesterol", f"{chol} mg/dL - borderline high", -0.06)
    else:
        add("Cholesterol", f"{chol} mg/dL - desirable", 0.05)

    sys_bp, dia_bp = patient.get("Systolic_BP") or 120, patient.get("Diastolic_BP") or 80
    if sys_bp >= 140 or dia_bp >= 90:
        add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - hypertensive", -0.16)
    elif sys_bp >= 130 or dia_bp >= 80:
        add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - stage 1 range", -0.08)
    else:
        add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - normal range", 0.06)

    activity = patient.get("Physical_Activity_Hours_Week") or 0
    add("Physical Activity", f"{activity} hrs/week",
        min(0.15, activity * 0.03) if activity >= 2.5 else -0.05)

    sleep = patient.get("Sleep_Hours_Per_Night") or 7
    if 7 <= sleep <= 9:
        add("Sleep Duration", f"{sleep} hrs/night - optimal", 0.08)
    elif sleep < 6:
        add("Sleep Duration", f"{sleep} hrs/night - deficient", -0.10)
    else:
        add("Sleep Duration", f"{sleep} hrs/night - acceptable", 0.02)

    add("Diet Quality", patient.get("Diet_Quality", "Average"),
        {"Good": 0.10, "Average": 0.0, "Poor": -0.14}.get(patient.get("Diet_Quality"), 0.0))
    add("Smoking Status", patient.get("Smoking_Status", "Never"),
        {"Never": 0.08, "Former": -0.03, "Current": -0.22}.get(patient.get("Smoking_Status"), 0.0))
    add("Stress Level", patient.get("Stress_Level", "Moderate"),
        {"Low": 0.07, "Moderate": -0.02, "High": -0.16}.get(patient.get("Stress_Level"), 0.0))

    duration = patient.get("Treatment_Duration") or 30
    add("Treatment Duration", f"{duration} days of treatment", min(0.12, duration * 0.0015))

    age = patient.get("Age") or 45
    add("Age", f"{age} years", -0.002 * max(0, age - 40))

    probability = _sigmoid(logit)
    factors.sort(key=lambda f: f["magnitude"], reverse=True)
    return {"prediction": 1 if probability >= 0.5 else 0,
            "probability": round(probability, 4), "factors": factors, "source": "offline"}


# ===========================================================================
# SECTION C: Flask API client  (database.db persistence layer)
# ===========================================================================
def api_ping(base_url):
    try:
        return requests.get(f"{base_url.rstrip('/')}/", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def api_save_patient(base_url, patient, result):
    payload = {**patient, "prediction": result["prediction"],
               "probability": result["probability"], "factors": result.get("factors", []),
               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    resp = requests.post(f"{base_url.rstrip('/')}/patients", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def api_get_patients(base_url):
    resp = requests.get(f"{base_url.rstrip('/')}/patients", timeout=5)
    resp.raise_for_status()
    return resp.json()


def api_delete_patient(base_url, patient_id):
    requests.delete(f"{base_url.rstrip('/')}/patients/{patient_id}", timeout=5)


def api_clear_patients(base_url):
    requests.delete(f"{base_url.rstrip('/')}/patients", timeout=5)


def generate_sample_patient(rng, index):
    """Generate one varied demo patient for analytics/testing."""
    age = rng.randint(18, 82)
    gender = rng.choice(["Male", "Female"])
    living_area = rng.choice(["Urban", "Urban", "Rural", "Semi-Urban"])

    diabetes = 1 if rng.random() < (0.08 + max(0, age - 35) * 0.006) else 0
    hypertension = 1 if rng.random() < (0.10 + max(0, age - 30) * 0.009) else 0
    heart_disease = 1 if rng.random() < (0.04 + max(0, age - 40) * 0.004 + 0.05 * diabetes) else 0

    systolic = rng.randint(105, 175) if hypertension else rng.randint(105, 135)
    diastolic = rng.randint(65, 105) if hypertension else rng.randint(65, 88)
    glucose = round(rng.uniform(80, 115) + (45 if diabetes else 0) + rng.uniform(-8, 15), 1)
    cholesterol = round(rng.uniform(145, 225) + (25 if heart_disease else 0), 1)
    bmi = round(rng.uniform(18.5, 34.5), 1)

    past_options = [
        "None", "Arthritis", "Kidney Disease", "Asthma",
        "Thyroid Disorder", "Liver Disease", "Stroke History",
    ]
    past_disease = rng.choice(past_options)
    if heart_disease and rng.random() < 0.45:
        past_disease = "Heart Disease"
    elif diabetes and rng.random() < 0.35:
        past_disease = "Diabetes"

    activity = round(rng.uniform(0.5, 7.0), 1)
    sleep = round(rng.uniform(5.0, 9.0), 1)
    diet = rng.choices(["Good", "Average", "Poor"], weights=[25, 50, 25], k=1)[0]
    smoking = rng.choices(["Never", "Former", "Current"], weights=[55, 20, 25], k=1)[0]
    stress = rng.choices(["Low", "Moderate", "High"], weights=[25, 50, 25], k=1)[0]
    treatment = rng.choice(["Medication", "Lifestyle Changes", "Surgery", "Physiotherapy", "Combined Therapy"])
    duration = rng.randint(7, 240)
    follow_up = rng.choices(["Yes", "No"], weights=[78, 22], k=1)[0]

    return {
        **DEFAULT_PATIENT,
        "Age": age, "Gender": gender, "Living_Area": living_area,
        "Systolic_BP": systolic, "Diastolic_BP": diastolic,
        "Glucose": glucose, "Cholesterol": cholesterol, "BMI": bmi,
        "Diabetes": diabetes, "Hypertension": hypertension, "Heart_Disease": heart_disease,
        "Past_Diseases": past_disease,
        "Physical_Activity_Hours_Week": activity, "Sleep_Hours_Per_Night": sleep,
        "Diet_Quality": diet, "Smoking_Status": smoking, "Stress_Level": stress,
        "Treatment": treatment, "Treatment_Duration": duration, "Follow_Up": follow_up,
    }


def load_sample_dataset(base_url, count=100):
    """Create and persist a reproducible demo dataset through the normal API."""
    rng = random.Random(20260928)
    saved = 0
    for i in range(count):
        patient = generate_sample_patient(rng, i + 1)
        result = predict_offline(patient)
        # Keep sample records distributed across recent dates for time-series charts.
        payload_result = dict(result)
        saved_record = api_save_patient(base_url, patient, payload_result)
        saved += 1
    return saved


# ===========================================================================
# SECTION D: Session state
# ===========================================================================
defaults = {
    "patient": copy.deepcopy(DEFAULT_PATIENT), "result": None, "is_saved": False,
    "active_persona_id": None, "api_url": API_BASE_URL_DEFAULT, "backend_online": False,
    "form_section": "Vitals & Labs", "active_tab": "Dashboard",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def reset_form():
    st.session_state.patient = copy.deepcopy(DEFAULT_PATIENT)
    st.session_state.result = None
    st.session_state.is_saved = False
    st.session_state.active_persona_id = None


def apply_persona(persona):
    st.session_state.patient = copy.deepcopy(persona["patient"])
    st.session_state.active_persona_id = persona["id"]
    st.session_state.result = None
    st.session_state.is_saved = False


def fetch_history_df():
    """Pulls patient history from the Flask/SQLite backend into a DataFrame.
    Returns (df, backend_reachable)."""
    try:
        records = api_get_patients(st.session_state.api_url)
        st.session_state.backend_online = True
        return pd.DataFrame(records), True
    except requests.RequestException:
        st.session_state.backend_online = False
        return pd.DataFrame(), False


# ===========================================================================
# SECTION E: Left sidebar navigation + header
# ===========================================================================
NAV_ITEMS = [
    ("Dashboard", "📊"),
    ("New Prediction", "✨"),
    ("Patient History", "📜"),
    ("Disease Analysis", "🦠"),
    ("Treatment Analysis", "💊"),
    ("Risk Analysis", "⚠️"),
]


def render_sidebar_nav():
    with st.sidebar:
        st.markdown("## 💓 Health Predictor")
        # st.caption("`ML Model v1.0` — Clinical Decision Support")
        st.divider()

        labels = [f"{icon}  {name}" for name, icon in NAV_ITEMS]
        names = [name for name, _ in NAV_ITEMS]
        current_index = names.index(st.session_state.active_tab) if st.session_state.active_tab in names else 0

        selected_label = st.radio(
            "Navigation", labels, index=current_index, label_visibility="collapsed"
        )
        selected_name = names[labels.index(selected_label)]
        st.session_state.active_tab = selected_name

        st.divider()
        st.markdown("### 🧪 Demo Dataset")
        sample_count = st.selectbox("Sample records", [50, 75, 100], index=2)
        if st.button("📥 Load Sample Data", use_container_width=True):
            if not api_ping(st.session_state.api_url):
                st.error("Backend is not reachable. Start `flask_backend.py` first.")
            else:
                progress = st.progress(0, text="Loading sample patients...")
                try:
                    # Save in small batches so the UI remains responsive enough for a demo.
                    rng = random.Random(20260928)
                    for i in range(sample_count):
                        patient = generate_sample_patient(rng, i + 1)
                        result = predict_offline(patient)
                        api_save_patient(st.session_state.api_url, patient, result)
                        progress.progress((i + 1) / sample_count, text=f"Loading patient {i + 1}/{sample_count}")
                    progress.empty()
                    st.session_state.backend_online = True
                    st.success(f"Loaded {sample_count} sample patient records.")
                    st.rerun()
                except requests.RequestException as exc:
                    progress.empty()
                    st.error(f"Could not finish loading sample data: {exc}")

        st.caption("Use this demo data to populate Dashboard and all analysis tabs.")
        st.divider()
        with st.expander("⚙️ Backend settings"):
            st.session_state.api_url = st.text_input("Flask API base URL", value=st.session_state.api_url)
            if st.button("🔄 Ping backend"):
                st.session_state.backend_online = api_ping(st.session_state.api_url)
            st.caption("🟢 Online" if st.session_state.backend_online else "🟠 Not reachable yet")

    return selected_name



# ===========================================================================
# SECTION F: Dashboard tab
# ===========================================================================
def render_dashboard():
    st.subheader("📊 Dashboard")
    df, online = fetch_history_df()
    if not online:
        st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`, "
                   "then click **Backend settings → Ping backend** above.")
        return
    if df.empty:
        st.info("No assessments stored yet. Run a prediction in **New Prediction** and save it.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assessments", len(df))
    c2.metric("Improvement Rate", f"{round(df['prediction'].mean() * 100, 1)}%")
    c3.metric("Average Age", f"{round(df['age'].mean(), 1)} yrs")
    c4.metric("Average BMI", f"{round(df['bmi'].mean(), 1)}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Outcome Distribution**")
        outcome_counts = df["prediction"].map({1: "Improved", 0: "Unimproved"}).value_counts()
        st.bar_chart(outcome_counts)
    with col2:
        st.markdown("**Treatment Modality Distribution**")
        st.bar_chart(df["treatment"].value_counts())

    st.markdown("**Assessments Over Time**")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    st.line_chart(df.groupby("date").size().rename("Assessments"))


# ===========================================================================
# SECTION G: New Prediction tab (persona selector + form + result)
# ===========================================================================
def render_new_prediction():
    # -----------------------------------------------------------------------
    # NEW PREDICTION — clean two-panel clinical assessment workspace
    # -----------------------------------------------------------------------
    st.subheader("✨ New Patient Prediction")
    st.caption("Enter patient information systematically, run the assessment, and review the model result in the panel on the right.")

    # Quick test-case presets
    st.markdown("#### 🧪 Quick Patient Presets")
    preset_cols = st.columns(len(PATIENT_PERSONAS))
    for col, persona in zip(preset_cols, PATIENT_PERSONAS):
        with col:
            active = st.session_state.active_persona_id == persona["id"]
            if st.button(
                f"{'✅ ' if active else ''}{persona['name']}",
                key=f"persona_{persona['id']}",
                use_container_width=True,
            ):
                apply_persona(persona)
                st.rerun()
            st.caption(persona["tag"])

    top_a, top_b = st.columns([4, 1])
    with top_a:
        st.info("💡 Tip: complete the sections from left to right. The assessment panel stays visible while you enter the data.")
    with top_b:
        if st.button("↻ Reset Form", use_container_width=True):
            reset_form()
            st.rerun()

    st.divider()

    # Main workspace: larger input area + compact persistent result panel.
    left, right = st.columns([3.2, 2], gap="large")

    with left:
        p = st.session_state.patient

        st.markdown("### 📝 Patient Information")
        st.caption("Choose one section at a time to keep the assessment easy to scan.")

        sections = [
            "👤 Demographics",
            "🩺 Vitals & Labs",
            "📋 Medical History",
            "🏃 Lifestyle & Habits",
            "💊 Treatment & Care",
        ]
        current_label = next(
            (label for label in sections if label.endswith(st.session_state.form_section)),
            "🩺 Vitals & Labs",
        )
        selected_section = st.radio(
            "Assessment section",
            sections,
            index=sections.index(current_label),
            horizontal=True,
            label_visibility="collapsed",
            key="new_prediction_section_selector",
        )
        section = selected_section.split(" ", 1)[1]
        st.session_state.form_section = section

        with st.container(border=True):
            if section == "Demographics":
                st.markdown("#### 👤 Basic Demographics")
                c1, c2, c3 = st.columns(3)
                p["Age"] = c1.slider("Age (years)", 18, 100, int(p["Age"]))
                p["Gender"] = c2.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(p["Gender"]))
                area_options = ["Urban", "Suburban", "Rural"]
                p["Living_Area"] = c3.selectbox("Living Area", area_options, index=area_options.index(p["Living_Area"]))

            elif section == "Vitals & Labs":
                st.markdown("#### 🩺 Vitals & Laboratory Values")
                bp = getBPCategory(p["Systolic_BP"], p["Diastolic_BP"])
                st.markdown(f"{LEVEL_EMOJI[bp['level']]} **Blood Pressure:** {bp['label']}")
                c1, c2 = st.columns(2)
                p["Systolic_BP"] = c1.slider("Systolic BP (mmHg)", 70, 200, int(p["Systolic_BP"]))
                p["Diastolic_BP"] = c2.slider("Diastolic BP (mmHg)", 40, 130, int(p["Diastolic_BP"]))

                c1, c2, c3 = st.columns(3)
                with c1:
                    p["Glucose"] = st.number_input("Blood Glucose (mg/dL)", 50, 300, p["Glucose"])
                    category = getGlucoseCategory(p["Glucose"])
                    st.caption(f"{LEVEL_EMOJI[category['level']]} {category['label']}")
                with c2:
                    p["Cholesterol"] = st.number_input("Total Cholesterol (mg/dL)", 100, 400, p["Cholesterol"])
                    category = getCholesterolCategory(p["Cholesterol"])
                    st.caption(f"{LEVEL_EMOJI[category['level']]} {category['label']}")
                with c3:
                    p["BMI"] = st.number_input("BMI (kg/m²)", 14.0, 50.0, float(p["BMI"]), step=0.1)
                    category = getBMICategory(p["BMI"])
                    st.caption(f"{LEVEL_EMOJI[category['level']]} {category['label']}")

            elif section == "Medical History":
                st.markdown("#### 📋 Existing Conditions & History")
                c1, c2, c3 = st.columns(3)
                p["Diabetes"] = int(c1.toggle("Diabetes", value=bool(p["Diabetes"]), key="new_diabetes"))
                p["Hypertension"] = int(c2.toggle("Hypertension", value=bool(p["Hypertension"]), key="new_hypertension"))
                p["Heart_Disease"] = int(c3.toggle("Heart Disease", value=bool(p["Heart_Disease"]), key="new_heart_disease"))
                options = ["None", "Asthma", "Arthritis", "Kidney Disease", "Liver Disease", "Thyroid"]
                history_index = options.index(p["Past_Diseases"]) if p["Past_Diseases"] in options else 0
                p["Past_Diseases"] = st.selectbox("Primary Past Disease / Organ History", options, index=history_index)

            elif section == "Lifestyle & Habits":
                st.markdown("#### 🏃 Lifestyle & Daily Habits")
                c1, c2 = st.columns(2)
                p["Physical_Activity_Hours_Week"] = c1.slider(
                    "Physical Activity (hrs/week)", 0.0, 15.0,
                    float(p["Physical_Activity_Hours_Week"]), step=0.5,
                )
                p["Sleep_Hours_Per_Night"] = c2.slider(
                    "Sleep Duration (hrs/night)", 3.0, 12.0,
                    float(p["Sleep_Hours_Per_Night"]), step=0.5,
                )
                c1, c2, c3 = st.columns(3)
                p["Diet_Quality"] = c1.selectbox("Diet Quality", ["Good", "Average", "Poor"], index=["Good", "Average", "Poor"].index(p["Diet_Quality"]))
                p["Smoking_Status"] = c2.selectbox("Smoking Status", ["Never", "Former", "Current"], index=["Never", "Former", "Current"].index(p["Smoking_Status"]))
                p["Stress_Level"] = c3.selectbox("Stress Level", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(p["Stress_Level"]))

            elif section == "Treatment & Care":
                st.markdown("#### 💊 Treatment & Follow-Up")
                c1, c2 = st.columns(2)
                treatment_options = ["Medication", "Lifestyle Changes", "Surgery"]
                treatment_index = treatment_options.index(p["Treatment"]) if p["Treatment"] in treatment_options else 0
                p["Treatment"] = c1.selectbox("Treatment Modality", treatment_options, index=treatment_index)
                p["Treatment_Duration"] = c2.slider("Treatment Duration (days)", 1, 365, int(p["Treatment_Duration"]))
                p["Follow_Up"] = "Yes" if st.toggle("Scheduled follow-up attended", value=(p["Follow_Up"] == "Yes"), key="new_followup") else "No"

        st.session_state.patient = p

        st.markdown("#### 🔎 Assessment Readiness")
        checks = {
            "Demographics": bool(p.get("Age")) and bool(p.get("Gender")) and bool(p.get("Living_Area")),
            "Vitals & Labs": all(p.get(k) is not None for k in ["Systolic_BP", "Diastolic_BP", "Glucose", "Cholesterol", "BMI"]),
            "Medical History": all(k in p for k in ["Diabetes", "Hypertension", "Heart_Disease", "Past_Diseases"]),
            "Lifestyle & Habits": all(k in p for k in ["Physical_Activity_Hours_Week", "Sleep_Hours_Per_Night", "Diet_Quality", "Smoking_Status", "Stress_Level"]),
            "Treatment & Care": all(k in p for k in ["Treatment", "Treatment_Duration", "Follow_Up"]),
        }
        ready_count = sum(checks.values())
        r1, r2 = st.columns([3, 1])
        with r1:
            st.progress(ready_count / len(checks), text=f"{ready_count}/{len(checks)} assessment sections ready")
        with r2:
            st.caption("Ready to assess" if ready_count == len(checks) else "Complete the form")

        if st.button("✨ Run Patient Assessment", type="primary", use_container_width=True):
            st.session_state.result = predict_offline(st.session_state.patient)
            st.session_state.is_saved = False
            st.rerun()

    with right:
        st.markdown("### 📌 Assessment Panel")
        result = st.session_state.result

        if not result:
            with st.container(border=True):
                st.markdown("#### ⏳ Waiting for assessment")
                st.write("Your prediction will appear here after you click **Run Patient Assessment**.")
                st.markdown("**What will be shown:**")
                st.markdown("- Outcome status\n- Improvement probability\n- Key contributing factors\n- Clinical recommendations\n- Save-to-database action")
        else:
            is_improved = result["prediction"] == 1
            percent = round(result["probability"] * 1000) / 10
            status_text = "Likely to Improve" if is_improved else "Lower Improvement Probability"

            with st.container(border=True):
                st.markdown("#### 🎯 Model Outcome")
                if is_improved:
                    st.success(f"**{status_text}**")
                else:
                    st.warning(f"**{status_text}**")

                st.metric("Improvement Probability", f"{percent:.1f}%")
                st.progress(min(1.0, max(0.03, percent / 100)))

                c1, c2 = st.columns(2)
                c1.metric("Prediction", "Positive" if is_improved else "Guarded")
                c2.metric("Age", f"{p['Age']} yrs")

            with st.container(border=True):
                st.markdown("#### 🩺 Patient Snapshot")
                snapshot = [
                    ("BP", f"{p['Systolic_BP']}/{p['Diastolic_BP']} mmHg"),
                    ("Glucose", f"{p['Glucose']} mg/dL"),
                    ("Cholesterol", f"{p['Cholesterol']} mg/dL"),
                    ("BMI", f"{p['BMI']:.1f}"),
                    ("Diabetes", "Yes" if p["Diabetes"] else "No"),
                    ("Hypertension", "Yes" if p["Hypertension"] else "No"),
                ]
                for i in range(0, len(snapshot), 2):
                    c1, c2 = st.columns(2)
                    c1.caption(snapshot[i][0])
                    c1.markdown(f"**{snapshot[i][1]}**")
                    if i + 1 < len(snapshot):
                        c2.caption(snapshot[i + 1][0])
                        c2.markdown(f"**{snapshot[i + 1][1]}**")

            with st.container(border=True):
                st.markdown("#### 💾 Record")
                if st.session_state.is_saved:
                    st.success("Saved to database.db")
                else:
                    if st.button("💾 Save Assessment", type="secondary", use_container_width=True):
                        try:
                            api_save_patient(st.session_state.api_url, st.session_state.patient, result)
                            st.session_state.is_saved = True
                            st.rerun()
                        except requests.RequestException as e:
                            st.error(f"Could not reach Flask backend: {e}")

            st.caption("This is a model-generated assessment for the project interface and should not be treated as a medical diagnosis.")

    # Full-width explainability/advice area appears only after a prediction.
    if st.session_state.result:
        st.divider()
        st.markdown("### 📊 Explainability & Recommendations")
        c1, c2 = st.columns([1.15, 1])
        with c1:
            render_factor_impact(st.session_state.result)
        with c2:
            render_clinical_advice(st.session_state.patient, st.session_state.result)
        render_report_section(st.session_state.patient, st.session_state.result)

def render_factor_impact(result):
    factors = result.get("factors", [])
    if not factors:
        return
    st.markdown("##### 🔎 Explainable AI • Feature Contributions")
    filter_choice = st.radio("Filter", ["All (top 8)", "Protective only", "Risk drivers only"],
                              horizontal=True, label_visibility="collapsed", key="factor_filter")
    if filter_choice == "Protective only":
        shown = [f for f in factors if f["isPositive"]]
    elif filter_choice == "Risk drivers only":
        shown = [f for f in factors if not f["isPositive"]]
    else:
        shown = factors[:8]
    df = pd.DataFrame([{"Factor": f["name"], "Impact": f["impact"]} for f in shown]).set_index("Factor")
    st.bar_chart(df["Impact"])
    for f in shown:
        icon = "📈" if f["isPositive"] else "📉"
        st.caption(f"{icon} **{f['name']}** ({f['description']}) — "
                   f"{'+' if f['isPositive'] else ''}{f['impact']}")


def render_clinical_advice(patient, result):
    st.markdown("##### 🩺 Tailored Clinical Recommendations")
    recs = generateClinicalRecommendations(patient, result["probability"])
    badge = {"high": "🔴 High Priority", "medium": "🟠 Moderate Priority", "low": "🟢 Maintenance"}
    for rec in recs:
        with st.container(border=True):
            st.markdown(f"**{rec['title']}**  —  {badge.get(rec['priority'], '')}")
            st.caption(rec["detail"])


def render_report_section(patient, result):
    is_improved = result["prediction"] == 1
    percent = round(result["probability"] * 1000) / 10
    recs = generateClinicalRecommendations(patient, result["probability"])
    with st.expander("🖨️ Full Clinical Report (printable summary)"):
        lines = [
            "# PulsePredict Clinical Assessment",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            f"## Model Prediction: {'Favorable' if is_improved else 'Guarded'} Prognosis",
            f"Improvement Probability: **{percent}%**", "",
            "## Patient Parameters",
            f"- Demographics: {patient['Age']} yrs • {patient['Gender']} • {patient['Living_Area']}",
            f"- Blood Pressure: {patient['Systolic_BP']}/{patient['Diastolic_BP']} mmHg",
            f"- Glucose: {patient['Glucose']} mg/dL • Cholesterol: {patient['Cholesterol']} mg/dL • BMI: {patient['BMI']}",
            f"- Treatment: {patient['Treatment']} ({patient['Treatment_Duration']} days) • Follow-Up: {patient['Follow_Up']}",
            "", "## Recommendations",
        ] + [f"- **{r['title']}**: {r['detail']}" for r in recs]
        report_md = "\n".join(lines)
        st.markdown(report_md)
        st.download_button("⬇ Download Report (Markdown)", data=report_md,
                           file_name=f"PulsePredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                           mime="text/markdown")


# ===========================================================================
# SECTION H: Patient History tab (reads database.db via Flask)
# ===========================================================================
def render_patient_history():
    st.subheader("📜 Patient History")
    df, online = fetch_history_df()
    if not online:
        st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
        return
    if df.empty:
        st.info("No assessments saved yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    gender_filter = c1.multiselect("Filter by gender", sorted(df["gender"].unique()))
    treatment_filter = c2.multiselect("Filter by treatment", sorted(df["treatment"].unique()))
    outcome_filter = c3.selectbox("Filter by outcome", ["All", "Improved", "Unimproved"])
    sort_col = c4.selectbox("Sort by", ["id", "timestamp", "age", "probability"], index=0)

    filtered = df.copy()
    if gender_filter:
        filtered = filtered[filtered["gender"].isin(gender_filter)]
    if treatment_filter:
        filtered = filtered[filtered["treatment"].isin(treatment_filter)]
    if outcome_filter != "All":
        filtered = filtered[filtered["prediction"] == (1 if outcome_filter == "Improved" else 0)]

    # Ascending order (oldest / lowest first) as requested
    filtered = filtered.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    st.dataframe(
        filtered[["id", "timestamp", "age", "gender", "living_area", "systolic_bp", "diastolic_bp",
                  "glucose", "treatment", "prediction", "probability"]],
        use_container_width=True, hide_index=True,
    )

    st.caption("Select a record ID below to load it back into the New Prediction form, or delete it.")
    c1, c2, c3 = st.columns(3)
    selected_id = c1.number_input("Record ID", min_value=0, step=1, value=0)
    if c2.button("↪ Load into form") and selected_id in filtered["id"].values:
        row = filtered[filtered["id"] == selected_id].iloc[0]
        st.session_state.patient = {
            "Age": int(row["age"]), "Gender": row["gender"], "Living_Area": row["living_area"],
            "Systolic_BP": int(row["systolic_bp"]), "Diastolic_BP": int(row["diastolic_bp"]),
            "Glucose": row["glucose"], "Cholesterol": row["cholesterol"], "BMI": row["bmi"],
            "Diabetes": int(row["diabetes"]), "Hypertension": int(row["hypertension"]),
            "Heart_Disease": int(row["heart_disease"]), "Past_Diseases": row["past_diseases"],
            "Physical_Activity_Hours_Week": row["physical_activity"], "Sleep_Hours_Per_Night": row["sleep_hours"],
            "Diet_Quality": row["diet_quality"], "Smoking_Status": row["smoking_status"],
            "Stress_Level": row["stress_level"], "Treatment": row["treatment"],
            "Treatment_Duration": int(row["treatment_duration"]), "Follow_Up": row["follow_up"],
        }
        st.session_state.result = None
        st.success("Loaded. Switch to the **New Prediction** tab to view/edit it.")
    if c3.button("🗑 Delete record") and selected_id:
        api_delete_patient(st.session_state.api_url, int(selected_id))
        st.rerun()

    if st.button("🗑 Clear ALL history", type="secondary"):
        api_clear_patients(st.session_state.api_url)
        st.rerun()


# ===========================================================================
# SECTION I: Disease Analysis tab
# ===========================================================================
def _make_disease_pie_chart(pie_df, label):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    values = pie_df["Patients"].tolist()
    labels = pie_df.index.tolist()
    if sum(values) == 0:
        values = [1, 0]
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(label, fontsize=11)
    fig.tight_layout()
    return fig


def render_disease_analysis():
    st.subheader("🦠 Disease Analysis")
    st.caption("Disease-wise percentage, bar graph and pie chart from saved patient assessments.")

    df, online = fetch_history_df()
    if not online:
        st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
        return
    if df.empty:
        st.info("No data yet — save some assessments first to generate disease analytics.")
        return

    disease_cols = [
        ("diabetes", "Diabetes"),
        ("hypertension", "Hypertension"),
        ("heart_disease", "Heart Disease"),
    ]
    total = len(df)

    for col, _ in disease_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)

    st.markdown("### 📊 Disease-wise Analysis")

    # Every disease is displayed as one horizontal row:
    # Percentage | Bar Graph | Pie Chart
    for col, label in disease_cols:
        affected = int(df[col].sum())
        absent = total - affected
        prevalence = (affected / total * 100) if total else 0

        affected_df = df[df[col] == 1]
        improved_pct = affected_df["prediction"].mean() * 100 if not affected_df.empty else 0
        avg_probability = affected_df["probability"].mean() * 100 if not affected_df.empty else 0

        st.markdown(f"#### 🩺 {label}")
        pct_col, bar_col, pie_col = st.columns([0.85, 1.7, 1.2])

        with pct_col:
            st.metric("Prevalence", f"{prevalence:.1f}%")
            st.caption(f"{affected} of {total} patients")
            st.metric("Improved", f"{improved_pct:.1f}%")
            st.caption(f"Avg probability: {avg_probability:.1f}%")

        with bar_col:
            st.markdown("**Bar Graph — Present vs Absent**")
            bar_df = pd.DataFrame({
                "Group": ["Present", "Absent"],
                "Patients": [affected, absent],
            }).set_index("Group")
            st.bar_chart(bar_df, y="Patients", height=240)

        with pie_col:
            st.markdown("**Pie Chart — Distribution**")
            pie_df = pd.DataFrame({
                "Status": ["Present", "Absent"],
                "Patients": [affected, absent],
            }).set_index("Status")
            st.pyplot(_make_disease_pie_chart(pie_df, label), use_container_width=True)

        st.divider()

    # Compact comparison of all three diseases.
    st.markdown("### 📈 Disease Percentage Comparison")
    comparison = []
    for col, label in disease_cols:
        affected = int(df[col].sum())
        affected_df = df[df[col] == 1]
        comparison.append({
            "Disease": label,
            "Prevalence %": round((affected / total * 100), 1),
            "Improved %": round(affected_df["prediction"].mean() * 100, 1) if not affected_df.empty else 0,
            "Avg Probability %": round(affected_df["probability"].mean() * 100, 1) if not affected_df.empty else 0,
        })

    comparison_df = pd.DataFrame(comparison).set_index("Disease")
    st.dataframe(
        comparison_df.style.format("{:.1f}%"),
        use_container_width=True,
    )

    st.markdown("### 📋 Past Disease / Organ History")
    if "past_diseases" in df.columns:
        past = df["past_diseases"].fillna("None").astype(str).str.strip()
        past = past[past.str.lower().ne("none") & past.ne("")]

        if past.empty:
            st.info("No past disease / organ history has been recorded in saved assessments.")
        else:
            past_items = past.str.split(",").explode().astype(str).str.strip()
            past_items = past_items[past_items.ne("")]
            past_counts = past_items.value_counts().rename_axis("Past Disease").to_frame("Patients")
            past_counts["Percentage"] = (past_counts["Patients"] / total * 100).round(1)
            pc1, pc2 = st.columns([1.4, 1])
            with pc1:
                st.markdown("**Frequency of recorded past diseases**")
                st.bar_chart(past_counts["Patients"], height=300)
            with pc2:
                st.markdown("**Percentage**")
                st.dataframe(
                    past_counts.assign(Percentage=past_counts["Percentage"].map(lambda x: f"{x:.1f}%")),
                    use_container_width=True,
                )

    st.caption("Percentages are calculated from the saved patient assessments currently available in the backend database. This is descriptive analysis and does not establish medical causation.")


# ===========================================================================
# SECTION J: Treatment Analysis tab
# ===========================================================================
def render_treatment_analysis():
    st.subheader("💊 Treatment Analysis")
    df, online = fetch_history_df()
    if not online or df.empty:
        st.info("No data yet — save some assessments first.")
        return

    st.markdown("**Average improvement probability by treatment modality**")
    st.bar_chart(df.groupby("treatment")["probability"].mean())

    st.markdown("**Improvement rate: Follow-up attended vs. missed**")
    st.bar_chart(df.groupby("follow_up")["prediction"].mean())

    st.markdown("**Treatment duration vs. improvement probability**")
    st.scatter_chart(df, x="treatment_duration", y="probability", color="treatment")


# ===========================================================================
# SECTION K: Risk Analysis tab
# ===========================================================================
def render_risk_analysis():
    st.subheader("⚠️ Risk Analysis")
    df, online = fetch_history_df()
    if not online or df.empty:
        st.info("No data yet — save some assessments first.")
        return

    st.markdown("**Probability distribution across all assessments**")
    st.bar_chart(pd.cut(df["probability"], bins=10).value_counts().sort_index().rename(
        lambda i: str(i)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average probability by smoking status**")
        st.bar_chart(df.groupby("smoking_status")["probability"].mean())
    with c2:
        st.markdown("**Average probability by stress level**")
        st.bar_chart(df.groupby("stress_level")["probability"].mean())

    st.markdown("**BMI category breakdown**")
    bmi_cats = df["bmi"].apply(lambda b: getBMICategory(b)["label"])
    st.bar_chart(bmi_cats.value_counts())


# ===========================================================================
# Page assembly
# ===========================================================================
active = render_sidebar_nav()

st.markdown("## 💓 Patient Health Outcome Predictor")
st.caption("Clinical Health Outcome Prognosis & Decision Support")
st.divider()

if active == "Dashboard":
    render_dashboard()
elif active == "New Prediction":
    render_new_prediction()
elif active == "Patient History":
    render_patient_history()
elif active == "Disease Analysis":
    render_disease_analysis()
elif active == "Treatment Analysis":
    render_treatment_analysis()
elif active == "Risk Analysis":
    render_risk_analysis()
    
    
    
    
    
    # """
# streamlit_app.py
# -----------------
# PulsePredict - Patient Health Outcome Predictor (single-file Streamlit app)

# Everything the previous 3-file version split across clinical_data.py /
# predictor.py / app.py now lives in this one file, plus a new top navigation
# bar with 6 sections:

#     Dashboard | New Prediction | Patient History |
#     Disease Analysis | Treatment Analysis | Risk Analysis

# Persistence: every saved assessment is POSTed to a Flask backend
# (flask_backend.py) which stores it in database.db (SQLite). Patient
# History / Dashboard / the 3 analysis tabs all read from that database, so
# your data survives across Streamlit restarts -- not just session memory.

# Run (two terminals):
#     Terminal 1:  python flask_backend.py        # starts DB + API on :5000
#     Terminal 2:  streamlit run streamlit_app.py  # starts the UI
# """

# import copy
# import json
# import math
# from datetime import datetime

# import pandas as pd
# import requests
# import streamlit as st

# st.set_page_config(
#     page_title="PulsePredict - Patient Health Outcome Predictor",
#     page_icon="💓",
#     layout="wide",
# )

# API_BASE_URL_DEFAULT = "http://127.0.0.1:5000"


# # ===========================================================================
# # SECTION A: Clinical reference data  (formerly clinical_data.py)
# # ===========================================================================
# DEFAULT_PATIENT = {
#     "Age": 45, "Gender": "Male", "Living_Area": "Urban",
#     "Systolic_BP": 120, "Diastolic_BP": 80,
#     "Glucose": 100, "Cholesterol": 180, "BMI": 24.5,
#     "Diabetes": 0, "Hypertension": 0, "Heart_Disease": 0,
#     "Past_Diseases": "None",
#     "Physical_Activity_Hours_Week": 2.5, "Sleep_Hours_Per_Night": 7.0,
#     "Diet_Quality": "Average", "Smoking_Status": "Never", "Stress_Level": "Moderate",
#     "Treatment": "Medication", "Treatment_Duration": 30, "Follow_Up": "Yes",
# }

# LEVEL_EMOJI = {"good": "🟢", "watch": "🟡", "warn": "🟠", "bad": "🔴", "low": "🔵", "critical": "🟣"}


# def getBPCategory(systolic, diastolic):
#     systolic, diastolic = systolic or 0, diastolic or 0
#     if systolic < 90 or diastolic < 60:
#         return {"label": "Hypotension", "level": "low"}
#     if systolic < 120 and diastolic < 80:
#         return {"label": "Normal", "level": "good"}
#     if systolic < 130 and diastolic < 80:
#         return {"label": "Elevated", "level": "watch"}
#     if systolic < 140 or diastolic < 90:
#         return {"label": "Hypertension Stage 1", "level": "warn"}
#     if systolic < 180 and diastolic < 120:
#         return {"label": "Hypertension Stage 2", "level": "bad"}
#     return {"label": "Hypertensive Crisis", "level": "critical"}


# def getGlucoseCategory(glucose):
#     glucose = glucose or 0
#     if glucose < 70:
#         return {"label": "Hypoglycemia", "level": "warn"}
#     if glucose <= 99:
#         return {"label": "Normal", "level": "good"}
#     if glucose <= 125:
#         return {"label": "Prediabetic Range", "level": "watch"}
#     return {"label": "Diabetic Range", "level": "bad"}


# def getCholesterolCategory(chol):
#     chol = chol or 0
#     if chol < 200:
#         return {"label": "Desirable", "level": "good"}
#     if chol < 240:
#         return {"label": "Borderline High", "level": "watch"}
#     return {"label": "High", "level": "bad"}


# def getBMICategory(bmi):
#     bmi = bmi or 0
#     if bmi < 18.5:
#         return {"label": "Underweight", "level": "watch"}
#     if bmi < 25:
#         return {"label": "Normal", "level": "good"}
#     if bmi < 30:
#         return {"label": "Overweight", "level": "watch"}
#     return {"label": "Obese", "level": "bad"}


# PATIENT_PERSONAS = [
#     {"id": "optimal", "name": "Optimal Health Profile", "tag": "Low Risk",
#      "description": "Healthy vitals, active lifestyle, full adherence.",
#      "patient": {**DEFAULT_PATIENT, "Age": 34, "Systolic_BP": 115, "Diastolic_BP": 75,
#                  "Glucose": 88, "Cholesterol": 170, "BMI": 22.0, "Physical_Activity_Hours_Week": 6.0,
#                  "Sleep_Hours_Per_Night": 8.0, "Diet_Quality": "Good", "Stress_Level": "Low",
#                  "Treatment": "Lifestyle Changes", "Treatment_Duration": 20}},
#     {"id": "high-risk", "name": "High-Risk Comorbid Patient", "tag": "High Risk",
#      "description": "Multiple chronic conditions, poor lifestyle indicators.",
#      "patient": {**DEFAULT_PATIENT, "Age": 67, "Systolic_BP": 162, "Diastolic_BP": 98,
#                  "Glucose": 185, "Cholesterol": 265, "BMI": 33.5, "Diabetes": 1, "Hypertension": 1,
#                  "Heart_Disease": 1, "Past_Diseases": "Kidney Disease", "Physical_Activity_Hours_Week": 0.5,
#                  "Sleep_Hours_Per_Night": 5.0, "Diet_Quality": "Poor", "Smoking_Status": "Current",
#                  "Stress_Level": "High", "Treatment_Duration": 180, "Follow_Up": "No"}},
#     {"id": "post-surgical", "name": "Post-Surgical Recovery", "tag": "Monitoring",
#      "description": "Recent surgery, moderate vitals, good compliance.",
#      "patient": {**DEFAULT_PATIENT, "Age": 52, "Systolic_BP": 128, "Diastolic_BP": 84,
#                  "Glucose": 110, "Cholesterol": 205, "BMI": 26.0, "Hypertension": 1,
#                  "Past_Diseases": "Arthritis", "Physical_Activity_Hours_Week": 1.5,
#                  "Sleep_Hours_Per_Night": 6.5, "Smoking_Status": "Former",
#                  "Treatment": "Surgery", "Treatment_Duration": 14}},
#     {"id": "young-lifestyle", "name": "Young Adult, Lifestyle Risk", "tag": "Lifestyle Watch",
#      "description": "Young patient with modifiable lifestyle risk factors.",
#      "patient": {**DEFAULT_PATIENT, "Age": 26, "Systolic_BP": 124, "Diastolic_BP": 79,
#                  "Glucose": 98, "Cholesterol": 195, "BMI": 27.5, "Physical_Activity_Hours_Week": 1.0,
#                  "Sleep_Hours_Per_Night": 5.5, "Diet_Quality": "Poor", "Smoking_Status": "Current",
#                  "Stress_Level": "High", "Treatment": "Lifestyle Changes", "Treatment_Duration": 10}},
# ]


# def generateClinicalRecommendations(patient, probability):
#     recs = []
#     bp = getBPCategory(patient.get("Systolic_BP"), patient.get("Diastolic_BP"))
#     if bp["level"] in ("warn", "bad", "critical"):
#         recs.append({"title": "Blood Pressure Management",
#                       "detail": f"Classified as '{bp['label']}'. Recommend antihypertensive review and home BP monitoring.",
#                       "priority": "high" if bp["level"] in ("bad", "critical") else "medium"})
#     gl = getGlucoseCategory(patient.get("Glucose"))
#     if gl["level"] in ("watch", "bad"):
#         recs.append({"title": "Glycemic Control",
#                       "detail": f"Glucose in '{gl['label']}' range. Suggest HbA1c testing and dietary review.",
#                       "priority": "high" if gl["level"] == "bad" else "medium"})
#     chol = getCholesterolCategory(patient.get("Cholesterol"))
#     if chol["level"] in ("watch", "bad"):
#         recs.append({"title": "Lipid Profile Follow-up",
#                       "detail": f"Cholesterol is '{chol['label']}'. Consider statin evaluation and fasting lipid panel.",
#                       "priority": "medium" if chol["level"] == "watch" else "high"})
#     bmi = getBMICategory(patient.get("BMI"))
#     if bmi["level"] in ("watch", "bad"):
#         recs.append({"title": "Weight Management",
#                       "detail": f"BMI is '{bmi['label']}'. Recommend nutrition counseling and activity ramp-up.",
#                       "priority": "medium"})
#     if patient.get("Smoking_Status") == "Current":
#         recs.append({"title": "Smoking Cessation",
#                       "detail": "Active smoking impairs recovery. Offer cessation counseling / NRT.",
#                       "priority": "high"})
#     if (patient.get("Physical_Activity_Hours_Week") or 0) < 2.5:
#         recs.append({"title": "Increase Physical Activity",
#                       "detail": "Below WHO-recommended 2.5 hrs/week. Suggest graded activity program.",
#                       "priority": "medium"})
#     if (patient.get("Sleep_Hours_Per_Night") or 0) < 6:
#         recs.append({"title": "Sleep Hygiene",
#                       "detail": "Sleep below 7-9h range may delay recovery. Consider sleep hygiene consult.",
#                       "priority": "low"})
#     if patient.get("Stress_Level") == "High":
#         recs.append({"title": "Stress & Mental Health Support",
#                       "detail": "High stress linked to poorer adherence. Recommend counseling referral.",
#                       "priority": "medium"})
#     if patient.get("Follow_Up") != "Yes":
#         recs.append({"title": "Follow-Up Appointment Adherence",
#                       "detail": "No follow-up attended -- strongest positive predictor. Reschedule immediately.",
#                       "priority": "high"})
#     if patient.get("Diabetes") or patient.get("Hypertension") or patient.get("Heart_Disease"):
#         recs.append({"title": "Chronic Condition Co-Management",
#                       "detail": "Comorbidities present. Recommend coordinated specialist care.",
#                       "priority": "medium"})
#     if not recs:
#         recs.append({"title": "Maintain Current Regimen",
#                       "detail": "All major biomarkers within healthy ranges. Continue routine check-ups.",
#                       "priority": "low"})
#     order = {"high": 0, "medium": 1, "low": 2}
#     recs.sort(key=lambda r: order.get(r["priority"], 3))
#     return recs


# # ===========================================================================
# # SECTION B: Offline prediction engine  (formerly predictor.py)
# # ===========================================================================
# def _sigmoid(x):
#     return 1 / (1 + math.exp(-x))


# def predict_offline(patient):
#     """Transparent weighted-scoring stand-in for a trained ML model -- swap
#     this out for model.predict_proba(...) once you have a real trained
#     model.pkl; the rest of the app only cares about the returned dict shape."""
#     factors = []
#     logit = 0.0

#     def add(name, description, weight):
#         nonlocal logit
#         logit += weight
#         factors.append({"name": name, "description": description,
#                          "impact": round(weight, 3), "magnitude": abs(weight),
#                          "isPositive": weight > 0})

#     add("Follow-Up Adherence",
#         "Attended scheduled follow-up" if patient.get("Follow_Up") == "Yes" else "Missed follow-up appointment",
#         0.275 if patient.get("Follow_Up") == "Yes" else -0.275)

#     if patient.get("Diabetes"):
#         add("Diabetes", "Diagnosed diabetes mellitus", -0.22)
#     if patient.get("Hypertension"):
#         add("Hypertension", "Diagnosed hypertension", -0.15)
#     if patient.get("Heart_Disease"):
#         add("Heart Disease", "Diagnosed cardiac history", -0.30)

#     bmi = patient.get("BMI") or 24
#     if bmi < 18.5 or bmi >= 30:
#         add("BMI", f"BMI of {bmi} outside healthy range", -0.15)
#     elif bmi >= 25:
#         add("BMI", f"BMI of {bmi} slightly elevated", -0.07)
#     else:
#         add("BMI", f"BMI of {bmi} within healthy range", 0.08)

#     glucose = patient.get("Glucose") or 100
#     if glucose >= 126:
#         add("Blood Glucose", f"{glucose} mg/dL - diabetic range", -0.18)
#     elif glucose >= 100:
#         add("Blood Glucose", f"{glucose} mg/dL - prediabetic range", -0.08)
#     else:
#         add("Blood Glucose", f"{glucose} mg/dL - normal range", 0.07)

#     chol = patient.get("Cholesterol") or 180
#     if chol >= 240:
#         add("Cholesterol", f"{chol} mg/dL - high", -0.14)
#     elif chol >= 200:
#         add("Cholesterol", f"{chol} mg/dL - borderline high", -0.06)
#     else:
#         add("Cholesterol", f"{chol} mg/dL - desirable", 0.05)

#     sys_bp, dia_bp = patient.get("Systolic_BP") or 120, patient.get("Diastolic_BP") or 80
#     if sys_bp >= 140 or dia_bp >= 90:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - hypertensive", -0.16)
#     elif sys_bp >= 130 or dia_bp >= 80:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - stage 1 range", -0.08)
#     else:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - normal range", 0.06)

#     activity = patient.get("Physical_Activity_Hours_Week") or 0
#     add("Physical Activity", f"{activity} hrs/week",
#         min(0.15, activity * 0.03) if activity >= 2.5 else -0.05)

#     sleep = patient.get("Sleep_Hours_Per_Night") or 7
#     if 7 <= sleep <= 9:
#         add("Sleep Duration", f"{sleep} hrs/night - optimal", 0.08)
#     elif sleep < 6:
#         add("Sleep Duration", f"{sleep} hrs/night - deficient", -0.10)
#     else:
#         add("Sleep Duration", f"{sleep} hrs/night - acceptable", 0.02)

#     add("Diet Quality", patient.get("Diet_Quality", "Average"),
#         {"Good": 0.10, "Average": 0.0, "Poor": -0.14}.get(patient.get("Diet_Quality"), 0.0))
#     add("Smoking Status", patient.get("Smoking_Status", "Never"),
#         {"Never": 0.08, "Former": -0.03, "Current": -0.22}.get(patient.get("Smoking_Status"), 0.0))
#     add("Stress Level", patient.get("Stress_Level", "Moderate"),
#         {"Low": 0.07, "Moderate": -0.02, "High": -0.16}.get(patient.get("Stress_Level"), 0.0))

#     duration = patient.get("Treatment_Duration") or 30
#     add("Treatment Duration", f"{duration} days of treatment", min(0.12, duration * 0.0015))

#     age = patient.get("Age") or 45
#     add("Age", f"{age} years", -0.002 * max(0, age - 40))

#     probability = _sigmoid(logit)
#     factors.sort(key=lambda f: f["magnitude"], reverse=True)
#     return {"prediction": 1 if probability >= 0.5 else 0,
#             "probability": round(probability, 4), "factors": factors, "source": "offline"}


# # ===========================================================================
# # SECTION C: Flask API client  (database.db persistence layer)
# # ===========================================================================
# def api_ping(base_url):
#     try:
#         return requests.get(f"{base_url.rstrip('/')}/", timeout=2).status_code == 200
#     except requests.RequestException:
#         return False


# def api_save_patient(base_url, patient, result):
#     payload = {**patient, "prediction": result["prediction"],
#                "probability": result["probability"], "factors": result.get("factors", []),
#                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#     resp = requests.post(f"{base_url.rstrip('/')}/patients", json=payload, timeout=5)
#     resp.raise_for_status()
#     return resp.json()


# def api_get_patients(base_url):
#     resp = requests.get(f"{base_url.rstrip('/')}/patients", timeout=5)
#     resp.raise_for_status()
#     return resp.json()


# def api_delete_patient(base_url, patient_id):
#     requests.delete(f"{base_url.rstrip('/')}/patients/{patient_id}", timeout=5)


# def api_clear_patients(base_url):
#     requests.delete(f"{base_url.rstrip('/')}/patients", timeout=5)


# # ===========================================================================
# # SECTION D: Session state
# # ===========================================================================
# defaults = {
#     "patient": copy.deepcopy(DEFAULT_PATIENT), "result": None, "is_saved": False,
#     "active_persona_id": None, "api_url": API_BASE_URL_DEFAULT, "backend_online": False,
#     "form_section": "Vitals & Labs", "active_tab": "Dashboard",
# }
# for k, v in defaults.items():
#     st.session_state.setdefault(k, v)


# def reset_form():
#     st.session_state.patient = copy.deepcopy(DEFAULT_PATIENT)
#     st.session_state.result = None
#     st.session_state.is_saved = False
#     st.session_state.active_persona_id = None


# def apply_persona(persona):
#     st.session_state.patient = copy.deepcopy(persona["patient"])
#     st.session_state.active_persona_id = persona["id"]
#     st.session_state.result = None
#     st.session_state.is_saved = False


# def fetch_history_df():
#     """Pulls patient history from the Flask/SQLite backend into a DataFrame.
#     Returns (df, backend_reachable)."""
#     try:
#         records = api_get_patients(st.session_state.api_url)
#         st.session_state.backend_online = True
#         return pd.DataFrame(records), True
#     except requests.RequestException:
#         st.session_state.backend_online = False
#         return pd.DataFrame(), False


# # ===========================================================================
# # SECTION E: Top navigation bar + header
# # ===========================================================================
# def render_top_bar():
#     c1, c2 = st.columns([3, 2])
#     with c1:
#         st.markdown("## 💓 PulsePredict  `ML Model v1.0`")
#         st.caption("Clinical Health Outcome Prognosis & Decision Support")
#     with c2:
#         with st.popover("⚙️ Backend settings"):
#             st.session_state.api_url = st.text_input("Flask API base URL", value=st.session_state.api_url)
#             if st.button("🔄 Ping backend"):
#                 st.session_state.backend_online = api_ping(st.session_state.api_url)
#             st.caption("🟢 Online" if st.session_state.backend_online else "🟠 Not reachable yet")

#     NAV_TABS = ["Dashboard", "New Prediction", "Patient History",
#                 "Disease Analysis", "Treatment Analysis", "Risk Analysis"]
#     tabs = st.tabs([f"  {t}  " for t in NAV_TABS])
#     return dict(zip(NAV_TABS, tabs))


# # ===========================================================================
# # SECTION F: Dashboard tab
# # ===========================================================================
# def render_dashboard():
#     st.subheader("📊 Dashboard")
#     df, online = fetch_history_df()
#     if not online:
#         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`, "
#                    "then click **Backend settings → Ping backend** above.")
#         return
#     if df.empty:
#         st.info("No assessments stored yet. Run a prediction in **New Prediction** and save it.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Assessments", len(df))
#     c2.metric("Improvement Rate", f"{round(df['prediction'].mean() * 100, 1)}%")
#     c3.metric("Average Age", f"{round(df['age'].mean(), 1)} yrs")
#     c4.metric("Average BMI", f"{round(df['bmi'].mean(), 1)}")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**Outcome Distribution**")
#         outcome_counts = df["prediction"].map({1: "Improved", 0: "Unimproved"}).value_counts()
#         st.bar_chart(outcome_counts)
#     with col2:
#         st.markdown("**Treatment Modality Distribution**")
#         st.bar_chart(df["treatment"].value_counts())

#     st.markdown("**Assessments Over Time**")
#     df["date"] = pd.to_datetime(df["timestamp"]).dt.date
#     st.line_chart(df.groupby("date").size().rename("Assessments"))


# # ===========================================================================
# # SECTION G: New Prediction tab (persona selector + form + result)
# # ===========================================================================
# def render_new_prediction():
#     st.markdown("##### Clinical Persona Presets  _(quick 1-click test cases)_")
#     cols = st.columns(len(PATIENT_PERSONAS))
#     for col, persona in zip(cols, PATIENT_PERSONAS):
#         with col:
#             is_active = st.session_state.active_persona_id == persona["id"]
#             if st.button(f"{'✅ ' if is_active else ''}**{persona['name']}**",
#                          key=f"persona_{persona['id']}", use_container_width=True):
#                 apply_persona(persona)
#                 st.rerun()
#             st.caption(f"`{persona['tag']}` — {persona['description']}")

#     if st.button("↻ Reset Form"):
#         reset_form()
#         st.rerun()

#     st.divider()
#     left, right = st.columns([3, 2])

#     with left:
#         p = st.session_state.patient
#         st.session_state.form_section = st.radio(
#             "Section", ["Vitals & Labs", "Medical History", "Lifestyle & Habits",
#                         "Treatment & Care", "Demographics"],
#             horizontal=True, label_visibility="collapsed",
#         )
#         section = st.session_state.form_section

#         if section == "Vitals & Labs":
#             bp = getBPCategory(p["Systolic_BP"], p["Diastolic_BP"])
#             st.markdown(f"{LEVEL_EMOJI[bp['level']]} **Blood Pressure status: {bp['label']}**")
#             c1, c2 = st.columns(2)
#             p["Systolic_BP"] = c1.slider("Systolic BP (mmHg)", 70, 200, p["Systolic_BP"])
#             p["Diastolic_BP"] = c2.slider("Diastolic BP (mmHg)", 40, 130, p["Diastolic_BP"])
#             c1, c2, c3 = st.columns(3)
#             with c1:
#                 p["Glucose"] = st.number_input("Blood Glucose (mg/dL)", 50, 300, p["Glucose"])
#                 st.caption(f"{LEVEL_EMOJI[getGlucoseCategory(p['Glucose'])['level']]} "
#                            f"{getGlucoseCategory(p['Glucose'])['label']}")
#             with c2:
#                 p["Cholesterol"] = st.number_input("Total Cholesterol (mg/dL)", 100, 400, p["Cholesterol"])
#                 st.caption(f"{LEVEL_EMOJI[getCholesterolCategory(p['Cholesterol'])['level']]} "
#                            f"{getCholesterolCategory(p['Cholesterol'])['label']}")
#             with c3:
#                 p["BMI"] = st.number_input("BMI (kg/m²)", 14.0, 50.0, float(p["BMI"]), step=0.1)
#                 st.caption(f"{LEVEL_EMOJI[getBMICategory(p['BMI'])['level']]} "
#                            f"{getBMICategory(p['BMI'])['label']}")

#         elif section == "Medical History":
#             c1, c2, c3 = st.columns(3)
#             p["Diabetes"] = int(c1.toggle("Diabetes Mellitus", value=bool(p["Diabetes"])))
#             p["Hypertension"] = int(c2.toggle("Hypertension", value=bool(p["Hypertension"])))
#             p["Heart_Disease"] = int(c3.toggle("Heart Disease", value=bool(p["Heart_Disease"])))
#             options = ["None", "Asthma", "Arthritis", "Kidney Disease", "Liver Disease", "Thyroid"]
#             p["Past_Diseases"] = st.radio("Primary Past Disease or Organ History", options,
#                                            index=options.index(p["Past_Diseases"]), horizontal=True)

#         elif section == "Lifestyle & Habits":
#             c1, c2 = st.columns(2)
#             p["Physical_Activity_Hours_Week"] = c1.slider("Physical Activity (hrs/week)", 0.0, 15.0,
#                                                            float(p["Physical_Activity_Hours_Week"]), step=0.5)
#             p["Sleep_Hours_Per_Night"] = c2.slider("Sleep Duration (hrs/night)", 3.0, 12.0,
#                                                    float(p["Sleep_Hours_Per_Night"]), step=0.5)
#             c1, c2, c3 = st.columns(3)
#             p["Diet_Quality"] = c1.radio("Diet Quality", ["Good", "Average", "Poor"],
#                                           index=["Good", "Average", "Poor"].index(p["Diet_Quality"]))
#             p["Smoking_Status"] = c2.radio("Smoking Status", ["Never", "Former", "Current"],
#                                             index=["Never", "Former", "Current"].index(p["Smoking_Status"]))
#             p["Stress_Level"] = c3.radio("Stress Level", ["Low", "Moderate", "High"],
#                                           index=["Low", "Moderate", "High"].index(p["Stress_Level"]))

#         elif section == "Treatment & Care":
#             c1, c2 = st.columns(2)
#             with c1:
#                 options = ["Medication", "Lifestyle Changes", "Surgery"]
#                 p["Treatment"] = st.radio("Treatment Modality", options, index=options.index(p["Treatment"]))
#             with c2:
#                 p["Treatment_Duration"] = st.slider("Treatment Duration (days)", 1, 365, p["Treatment_Duration"])
#                 p["Follow_Up"] = "Yes" if st.toggle("Attended Scheduled Follow-Up",
#                                                       value=(p["Follow_Up"] == "Yes")) else "No"

#         elif section == "Demographics":
#             c1, c2, c3 = st.columns(3)
#             p["Age"] = c1.slider("Age (years)", 18, 100, p["Age"])
#             p["Gender"] = c2.radio("Gender", ["Female", "Male"], index=["Female", "Male"].index(p["Gender"]))
#             options = ["Urban", "Suburban", "Rural"]
#             p["Living_Area"] = c3.radio("Living Area", options, index=options.index(p["Living_Area"]))

#         st.session_state.patient = p
#         st.divider()
#         if st.button("✨ Predict Patient Outcome", type="primary", use_container_width=True):
#             st.session_state.result = predict_offline(st.session_state.patient)
#             st.session_state.is_saved = False
#             st.rerun()

#     with right:
#         result = st.session_state.result
#         if not result:
#             st.info("⏳ Awaiting Patient Assessment — fill in the form, then click "
#                      "**Predict Patient Outcome**.")
#         else:
#             is_improved = result["prediction"] == 1
#             percent = round(result["probability"] * 1000) / 10
#             if is_improved:
#                 st.success("### ✅ Condition Likely to Improve")
#             else:
#                 st.error("### ⚠️ Guarded Prognosis / Low Improvement")
#             st.metric("Probability of Improvement", f"{percent}%")
#             st.progress(min(1.0, max(0.03, percent / 100)))

#             if st.session_state.is_saved:
#                 st.button("✅ Saved to Database", disabled=True, use_container_width=True)
#             else:
#                 if st.button("💾 Save to Database (database.db)", use_container_width=True):
#                     try:
#                         api_save_patient(st.session_state.api_url, st.session_state.patient, result)
#                         st.session_state.is_saved = True
#                         st.success("Saved to database.db via Flask backend.")
#                         st.rerun()
#                     except requests.RequestException as e:
#                         st.error(f"Could not reach Flask backend: {e}")

#     if st.session_state.result:
#         st.divider()
#         c1, c2 = st.columns(2)
#         with c1:
#             render_factor_impact(st.session_state.result)
#         with c2:
#             render_clinical_advice(st.session_state.patient, st.session_state.result)
#         render_report_section(st.session_state.patient, st.session_state.result)


# def render_factor_impact(result):
#     factors = result.get("factors", [])
#     if not factors:
#         return
#     st.markdown("##### 🔎 Explainable AI • Feature Contributions")
#     filter_choice = st.radio("Filter", ["All (top 8)", "Protective only", "Risk drivers only"],
#                               horizontal=True, label_visibility="collapsed", key="factor_filter")
#     if filter_choice == "Protective only":
#         shown = [f for f in factors if f["isPositive"]]
#     elif filter_choice == "Risk drivers only":
#         shown = [f for f in factors if not f["isPositive"]]
#     else:
#         shown = factors[:8]
#     df = pd.DataFrame([{"Factor": f["name"], "Impact": f["impact"]} for f in shown]).set_index("Factor")
#     st.bar_chart(df["Impact"])
#     for f in shown:
#         icon = "📈" if f["isPositive"] else "📉"
#         st.caption(f"{icon} **{f['name']}** ({f['description']}) — "
#                    f"{'+' if f['isPositive'] else ''}{f['impact']}")


# def render_clinical_advice(patient, result):
#     st.markdown("##### 🩺 Tailored Clinical Recommendations")
#     recs = generateClinicalRecommendations(patient, result["probability"])
#     badge = {"high": "🔴 High Priority", "medium": "🟠 Moderate Priority", "low": "🟢 Maintenance"}
#     for rec in recs:
#         with st.container(border=True):
#             st.markdown(f"**{rec['title']}**  —  {badge.get(rec['priority'], '')}")
#             st.caption(rec["detail"])


# def render_report_section(patient, result):
#     is_improved = result["prediction"] == 1
#     percent = round(result["probability"] * 1000) / 10
#     recs = generateClinicalRecommendations(patient, result["probability"])
#     with st.expander("🖨️ Full Clinical Report (printable summary)"):
#         lines = [
#             "# PulsePredict Clinical Assessment",
#             f"Date: {datetime.now().strftime('%Y-%m-%d')}",
#             f"## Model Prediction: {'Favorable' if is_improved else 'Guarded'} Prognosis",
#             f"Improvement Probability: **{percent}%**", "",
#             "## Patient Parameters",
#             f"- Demographics: {patient['Age']} yrs • {patient['Gender']} • {patient['Living_Area']}",
#             f"- Blood Pressure: {patient['Systolic_BP']}/{patient['Diastolic_BP']} mmHg",
#             f"- Glucose: {patient['Glucose']} mg/dL • Cholesterol: {patient['Cholesterol']} mg/dL • BMI: {patient['BMI']}",
#             f"- Treatment: {patient['Treatment']} ({patient['Treatment_Duration']} days) • Follow-Up: {patient['Follow_Up']}",
#             "", "## Recommendations",
#         ] + [f"- **{r['title']}**: {r['detail']}" for r in recs]
#         report_md = "\n".join(lines)
#         st.markdown(report_md)
#         st.download_button("⬇ Download Report (Markdown)", data=report_md,
#                            file_name=f"PulsePredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
#                            mime="text/markdown")


# # ===========================================================================
# # SECTION H: Patient History tab (reads database.db via Flask)
# # ===========================================================================
# def render_patient_history():
#     st.subheader("📜 Patient History")
#     df, online = fetch_history_df()
#     if not online:
#         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
#         return
#     if df.empty:
#         st.info("No assessments saved yet.")
#         return

#     c1, c2, c3 = st.columns(3)
#     gender_filter = c1.multiselect("Filter by gender", sorted(df["gender"].unique()))
#     treatment_filter = c2.multiselect("Filter by treatment", sorted(df["treatment"].unique()))
#     outcome_filter = c3.selectbox("Filter by outcome", ["All", "Improved", "Unimproved"])

#     filtered = df.copy()
#     if gender_filter:
#         filtered = filtered[filtered["gender"].isin(gender_filter)]
#     if treatment_filter:
#         filtered = filtered[filtered["treatment"].isin(treatment_filter)]
#     if outcome_filter != "All":
#         filtered = filtered[filtered["prediction"] == (1 if outcome_filter == "Improved" else 0)]

#     st.dataframe(
#         filtered[["id", "timestamp", "age", "gender", "living_area", "systolic_bp", "diastolic_bp",
#                   "glucose", "treatment", "prediction", "probability"]],
#         use_container_width=True, hide_index=True,
#     )

#     st.caption("Select a record ID below to load it back into the New Prediction form, or delete it.")
#     c1, c2, c3 = st.columns(3)
#     selected_id = c1.number_input("Record ID", min_value=0, step=1, value=0)
#     if c2.button("↪ Load into form") and selected_id in filtered["id"].values:
#         row = filtered[filtered["id"] == selected_id].iloc[0]
#         st.session_state.patient = {
#             "Age": int(row["age"]), "Gender": row["gender"], "Living_Area": row["living_area"],
#             "Systolic_BP": int(row["systolic_bp"]), "Diastolic_BP": int(row["diastolic_bp"]),
#             "Glucose": row["glucose"], "Cholesterol": row["cholesterol"], "BMI": row["bmi"],
#             "Diabetes": int(row["diabetes"]), "Hypertension": int(row["hypertension"]),
#             "Heart_Disease": int(row["heart_disease"]), "Past_Diseases": row["past_diseases"],
#             "Physical_Activity_Hours_Week": row["physical_activity"], "Sleep_Hours_Per_Night": row["sleep_hours"],
#             "Diet_Quality": row["diet_quality"], "Smoking_Status": row["smoking_status"],
#             "Stress_Level": row["stress_level"], "Treatment": row["treatment"],
#             "Treatment_Duration": int(row["treatment_duration"]), "Follow_Up": row["follow_up"],
#         }
#         st.session_state.result = None
#         st.success("Loaded. Switch to the **New Prediction** tab to view/edit it.")
#     if c3.button("🗑 Delete record") and selected_id:
#         api_delete_patient(st.session_state.api_url, int(selected_id))
#         st.rerun()

#     if st.button("🗑 Clear ALL history", type="secondary"):
#         api_clear_patients(st.session_state.api_url)
#         st.rerun()


# # ===========================================================================
# # SECTION I: Disease Analysis tab
# # ===========================================================================
# def render_disease_analysis():
#     st.subheader("🦠 Disease Analysis")
#     df, online = fetch_history_df()
#     if not online or df.empty:
#         st.info("No data yet — save some assessments first.")
#         return

#     for col, label in [("diabetes", "Diabetes"), ("hypertension", "Hypertension"), ("heart_disease", "Heart Disease")]:
#         st.markdown(f"**Improvement rate by {label} status**")
#         grouped = df.groupby(col)["prediction"].mean().rename(index={0: "No", 1: "Yes"})
#         st.bar_chart(grouped)

#     st.markdown("**Most common past disease / organ history**")
#     st.bar_chart(df["past_diseases"].value_counts())


# # ===========================================================================
# # SECTION J: Treatment Analysis tab
# # ===========================================================================
# def render_treatment_analysis():
#     st.subheader("💊 Treatment Analysis")
#     df, online = fetch_history_df()
#     if not online or df.empty:
#         st.info("No data yet — save some assessments first.")
#         return

#     st.markdown("**Average improvement probability by treatment modality**")
#     st.bar_chart(df.groupby("treatment")["probability"].mean())

#     st.markdown("**Improvement rate: Follow-up attended vs. missed**")
#     st.bar_chart(df.groupby("follow_up")["prediction"].mean())

#     st.markdown("**Treatment duration vs. improvement probability**")
#     st.scatter_chart(df, x="treatment_duration", y="probability", color="treatment")


# # ===========================================================================
# # SECTION K: Risk Analysis tab
# # ===========================================================================
# def render_risk_analysis():
#     st.subheader("⚠️ Risk Analysis")
#     df, online = fetch_history_df()
#     if not online or df.empty:
#         st.info("No data yet — save some assessments first.")
#         return

#     st.markdown("**Probability distribution across all assessments**")
#     st.bar_chart(pd.cut(df["probability"], bins=10).value_counts().sort_index().rename(
#         lambda i: str(i)))

#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("**Average probability by smoking status**")
#         st.bar_chart(df.groupby("smoking_status")["probability"].mean())
#     with c2:
#         st.markdown("**Average probability by stress level**")
#         st.bar_chart(df.groupby("stress_level")["probability"].mean())

#     st.markdown("**BMI category breakdown**")
#     bmi_cats = df["bmi"].apply(lambda b: getBMICategory(b)["label"])
#     st.bar_chart(bmi_cats.value_counts())


# # ===========================================================================
# # Page assembly
# # ===========================================================================
# tab_map = render_top_bar()

# with tab_map["Dashboard"]:
#     render_dashboard()
# with tab_map["New Prediction"]:
#     render_new_prediction()
# with tab_map["Patient History"]:
#     render_patient_history()
# with tab_map["Disease Analysis"]:
#     render_disease_analysis()
# with tab_map["Treatment Analysis"]:
#     render_treatment_analysis()
# with tab_map["Risk Analysis"]:
#     render_risk_analysis()






# """
# streamlit_app.py
# -----------------
# PulsePredict - Patient Health Outcome Predictor (single-file Streamlit app)

# Everything the previous 3-file version split across clinical_data.py /
# predictor.py / app.py now lives in this one file, plus a new top navigation
# bar with 6 sections:

#     Dashboard | New Prediction | Patient History |
#     Disease Analysis | Treatment Analysis | Risk Analysis

# Persistence: every saved assessment is POSTed to a Flask backend
# (flask_backend.py) which stores it in database.db (SQLite). Patient
# History / Dashboard / the 3 analysis tabs all read from that database, so
# your data survives across Streamlit restarts -- not just session memory.

# Run (two terminals):
#     Terminal 1:  python flask_backend.py        # starts DB + API on :5000
#     Terminal 2:  streamlit run streamlit_app.py  # starts the UI
# """

# import copy
# import json
# import math
# import random
# from datetime import datetime

# import pandas as pd
# import requests
# import streamlit as st

# st.set_page_config(
#     page_title="PulsePredict - Patient Health Outcome Predictor",
#     page_icon="💓",
#     layout="wide",
# )

# API_BASE_URL_DEFAULT = "http://127.0.0.1:5000"


# # ===========================================================================
# # SECTION A: Clinical reference data  (formerly clinical_data.py)
# # ===========================================================================
# DEFAULT_PATIENT = {
#     "Age": 45, "Gender": "Male", "Living_Area": "Urban",
#     "Systolic_BP": 120, "Diastolic_BP": 80,
#     "Glucose": 100, "Cholesterol": 180, "BMI": 24.5,
#     "Diabetes": 0, "Hypertension": 0, "Heart_Disease": 0,
#     "Past_Diseases": "None",
#     "Physical_Activity_Hours_Week": 2.5, "Sleep_Hours_Per_Night": 7.0,
#     "Diet_Quality": "Average", "Smoking_Status": "Never", "Stress_Level": "Moderate",
#     "Treatment": "Medication", "Treatment_Duration": 30, "Follow_Up": "Yes",
# }

# LEVEL_EMOJI = {"good": "🟢", "watch": "🟡", "warn": "🟠", "bad": "🔴", "low": "🔵", "critical": "🟣"}


# def getBPCategory(systolic, diastolic):
#     systolic, diastolic = systolic or 0, diastolic or 0
#     if systolic < 90 or diastolic < 60:
#         return {"label": "Hypotension", "level": "low"}
#     if systolic < 120 and diastolic < 80:
#         return {"label": "Normal", "level": "good"}
#     if systolic < 130 and diastolic < 80:
#         return {"label": "Elevated", "level": "watch"}
#     if systolic < 140 or diastolic < 90:
#         return {"label": "Hypertension Stage 1", "level": "warn"}
#     if systolic < 180 and diastolic < 120:
#         return {"label": "Hypertension Stage 2", "level": "bad"}
#     return {"label": "Hypertensive Crisis", "level": "critical"}


# def getGlucoseCategory(glucose):
#     glucose = glucose or 0
#     if glucose < 70:
#         return {"label": "Hypoglycemia", "level": "warn"}
#     if glucose <= 99:
#         return {"label": "Normal", "level": "good"}
#     if glucose <= 125:
#         return {"label": "Prediabetic Range", "level": "watch"}
#     return {"label": "Diabetic Range", "level": "bad"}


# def getCholesterolCategory(chol):
#     chol = chol or 0
#     if chol < 200:
#         return {"label": "Desirable", "level": "good"}
#     if chol < 240:
#         return {"label": "Borderline High", "level": "watch"}
#     return {"label": "High", "level": "bad"}


# def getBMICategory(bmi):
#     bmi = bmi or 0
#     if bmi < 18.5:
#         return {"label": "Underweight", "level": "watch"}
#     if bmi < 25:
#         return {"label": "Normal", "level": "good"}
#     if bmi < 30:
#         return {"label": "Overweight", "level": "watch"}
#     return {"label": "Obese", "level": "bad"}


# PATIENT_PERSONAS = [
#     {"id": "optimal", "name": "Optimal Health Profile", "tag": "Low Risk",
#      "description": "Healthy vitals, active lifestyle, full adherence.",
#      "patient": {**DEFAULT_PATIENT, "Age": 34, "Systolic_BP": 115, "Diastolic_BP": 75,
#                  "Glucose": 88, "Cholesterol": 170, "BMI": 22.0, "Physical_Activity_Hours_Week": 6.0,
#                  "Sleep_Hours_Per_Night": 8.0, "Diet_Quality": "Good", "Stress_Level": "Low",
#                  "Treatment": "Lifestyle Changes", "Treatment_Duration": 20}},
#     {"id": "high-risk", "name": "High-Risk Comorbid Patient", "tag": "High Risk",
#      "description": "Multiple chronic conditions, poor lifestyle indicators.",
#      "patient": {**DEFAULT_PATIENT, "Age": 67, "Systolic_BP": 162, "Diastolic_BP": 98,
#                  "Glucose": 185, "Cholesterol": 265, "BMI": 33.5, "Diabetes": 1, "Hypertension": 1,
#                  "Heart_Disease": 1, "Past_Diseases": "Kidney Disease", "Physical_Activity_Hours_Week": 0.5,
#                  "Sleep_Hours_Per_Night": 5.0, "Diet_Quality": "Poor", "Smoking_Status": "Current",
#                  "Stress_Level": "High", "Treatment_Duration": 180, "Follow_Up": "No"}},
#     {"id": "post-surgical", "name": "Post-Surgical Recovery", "tag": "Monitoring",
#      "description": "Recent surgery, moderate vitals, good compliance.",
#      "patient": {**DEFAULT_PATIENT, "Age": 52, "Systolic_BP": 128, "Diastolic_BP": 84,
#                  "Glucose": 110, "Cholesterol": 205, "BMI": 26.0, "Hypertension": 1,
#                  "Past_Diseases": "Arthritis", "Physical_Activity_Hours_Week": 1.5,
#                  "Sleep_Hours_Per_Night": 6.5, "Smoking_Status": "Former",
#                  "Treatment": "Surgery", "Treatment_Duration": 14}},
#     {"id": "young-lifestyle", "name": "Young Adult, Lifestyle Risk", "tag": "Lifestyle Watch",
#      "description": "Young patient with modifiable lifestyle risk factors.",
#      "patient": {**DEFAULT_PATIENT, "Age": 26, "Systolic_BP": 124, "Diastolic_BP": 79,
#                  "Glucose": 98, "Cholesterol": 195, "BMI": 27.5, "Physical_Activity_Hours_Week": 1.0,
#                  "Sleep_Hours_Per_Night": 5.5, "Diet_Quality": "Poor", "Smoking_Status": "Current",
#                  "Stress_Level": "High", "Treatment": "Lifestyle Changes", "Treatment_Duration": 10}},
# ]


# def generateClinicalRecommendations(patient, probability):
#     recs = []
#     bp = getBPCategory(patient.get("Systolic_BP"), patient.get("Diastolic_BP"))
#     if bp["level"] in ("warn", "bad", "critical"):
#         recs.append({"title": "Blood Pressure Management",
#                       "detail": f"Classified as '{bp['label']}'. Recommend antihypertensive review and home BP monitoring.",
#                       "priority": "high" if bp["level"] in ("bad", "critical") else "medium"})
#     gl = getGlucoseCategory(patient.get("Glucose"))
#     if gl["level"] in ("watch", "bad"):
#         recs.append({"title": "Glycemic Control",
#                       "detail": f"Glucose in '{gl['label']}' range. Suggest HbA1c testing and dietary review.",
#                       "priority": "high" if gl["level"] == "bad" else "medium"})
#     chol = getCholesterolCategory(patient.get("Cholesterol"))
#     if chol["level"] in ("watch", "bad"):
#         recs.append({"title": "Lipid Profile Follow-up",
#                       "detail": f"Cholesterol is '{chol['label']}'. Consider statin evaluation and fasting lipid panel.",
#                       "priority": "medium" if chol["level"] == "watch" else "high"})
#     bmi = getBMICategory(patient.get("BMI"))
#     if bmi["level"] in ("watch", "bad"):
#         recs.append({"title": "Weight Management",
#                       "detail": f"BMI is '{bmi['label']}'. Recommend nutrition counseling and activity ramp-up.",
#                       "priority": "medium"})
#     if patient.get("Smoking_Status") == "Current":
#         recs.append({"title": "Smoking Cessation",
#                       "detail": "Active smoking impairs recovery. Offer cessation counseling / NRT.",
#                       "priority": "high"})
#     if (patient.get("Physical_Activity_Hours_Week") or 0) < 2.5:
#         recs.append({"title": "Increase Physical Activity",
#                       "detail": "Below WHO-recommended 2.5 hrs/week. Suggest graded activity program.",
#                       "priority": "medium"})
#     if (patient.get("Sleep_Hours_Per_Night") or 0) < 6:
#         recs.append({"title": "Sleep Hygiene",
#                       "detail": "Sleep below 7-9h range may delay recovery. Consider sleep hygiene consult.",
#                       "priority": "low"})
#     if patient.get("Stress_Level") == "High":
#         recs.append({"title": "Stress & Mental Health Support",
#                       "detail": "High stress linked to poorer adherence. Recommend counseling referral.",
#                       "priority": "medium"})
#     if patient.get("Follow_Up") != "Yes":
#         recs.append({"title": "Follow-Up Appointment Adherence",
#                       "detail": "No follow-up attended -- strongest positive predictor. Reschedule immediately.",
#                       "priority": "high"})
#     if patient.get("Diabetes") or patient.get("Hypertension") or patient.get("Heart_Disease"):
#         recs.append({"title": "Chronic Condition Co-Management",
#                       "detail": "Comorbidities present. Recommend coordinated specialist care.",
#                       "priority": "medium"})
#     if not recs:
#         recs.append({"title": "Maintain Current Regimen",
#                       "detail": "All major biomarkers within healthy ranges. Continue routine check-ups.",
#                       "priority": "low"})
#     order = {"high": 0, "medium": 1, "low": 2}
#     recs.sort(key=lambda r: order.get(r["priority"], 3))
#     return recs


# # ===========================================================================
# # SECTION B: Offline prediction engine  (formerly predictor.py)
# # ===========================================================================
# def _sigmoid(x):
#     return 1 / (1 + math.exp(-x))


# def predict_offline(patient):
#     """Transparent weighted-scoring stand-in for a trained ML model -- swap
#     this out for model.predict_proba(...) once you have a real trained
#     model.pkl; the rest of the app only cares about the returned dict shape."""
#     factors = []
#     logit = 0.0

#     def add(name, description, weight):
#         nonlocal logit
#         logit += weight
#         factors.append({"name": name, "description": description,
#                          "impact": round(weight, 3), "magnitude": abs(weight),
#                          "isPositive": weight > 0})

#     add("Follow-Up Adherence",
#         "Attended scheduled follow-up" if patient.get("Follow_Up") == "Yes" else "Missed follow-up appointment",
#         0.275 if patient.get("Follow_Up") == "Yes" else -0.275)

#     if patient.get("Diabetes"):
#         add("Diabetes", "Diagnosed diabetes mellitus", -0.22)
#     if patient.get("Hypertension"):
#         add("Hypertension", "Diagnosed hypertension", -0.15)
#     if patient.get("Heart_Disease"):
#         add("Heart Disease", "Diagnosed cardiac history", -0.30)

#     bmi = patient.get("BMI") or 24
#     if bmi < 18.5 or bmi >= 30:
#         add("BMI", f"BMI of {bmi} outside healthy range", -0.15)
#     elif bmi >= 25:
#         add("BMI", f"BMI of {bmi} slightly elevated", -0.07)
#     else:
#         add("BMI", f"BMI of {bmi} within healthy range", 0.08)

#     glucose = patient.get("Glucose") or 100
#     if glucose >= 126:
#         add("Blood Glucose", f"{glucose} mg/dL - diabetic range", -0.18)
#     elif glucose >= 100:
#         add("Blood Glucose", f"{glucose} mg/dL - prediabetic range", -0.08)
#     else:
#         add("Blood Glucose", f"{glucose} mg/dL - normal range", 0.07)

#     chol = patient.get("Cholesterol") or 180
#     if chol >= 240:
#         add("Cholesterol", f"{chol} mg/dL - high", -0.14)
#     elif chol >= 200:
#         add("Cholesterol", f"{chol} mg/dL - borderline high", -0.06)
#     else:
#         add("Cholesterol", f"{chol} mg/dL - desirable", 0.05)

#     sys_bp, dia_bp = patient.get("Systolic_BP") or 120, patient.get("Diastolic_BP") or 80
#     if sys_bp >= 140 or dia_bp >= 90:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - hypertensive", -0.16)
#     elif sys_bp >= 130 or dia_bp >= 80:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - stage 1 range", -0.08)
#     else:
#         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - normal range", 0.06)

#     activity = patient.get("Physical_Activity_Hours_Week") or 0
#     add("Physical Activity", f"{activity} hrs/week",
#         min(0.15, activity * 0.03) if activity >= 2.5 else -0.05)

#     sleep = patient.get("Sleep_Hours_Per_Night") or 7
#     if 7 <= sleep <= 9:
#         add("Sleep Duration", f"{sleep} hrs/night - optimal", 0.08)
#     elif sleep < 6:
#         add("Sleep Duration", f"{sleep} hrs/night - deficient", -0.10)
#     else:
#         add("Sleep Duration", f"{sleep} hrs/night - acceptable", 0.02)

#     add("Diet Quality", patient.get("Diet_Quality", "Average"),
#         {"Good": 0.10, "Average": 0.0, "Poor": -0.14}.get(patient.get("Diet_Quality"), 0.0))
#     add("Smoking Status", patient.get("Smoking_Status", "Never"),
#         {"Never": 0.08, "Former": -0.03, "Current": -0.22}.get(patient.get("Smoking_Status"), 0.0))
#     add("Stress Level", patient.get("Stress_Level", "Moderate"),
#         {"Low": 0.07, "Moderate": -0.02, "High": -0.16}.get(patient.get("Stress_Level"), 0.0))

#     duration = patient.get("Treatment_Duration") or 30
#     add("Treatment Duration", f"{duration} days of treatment", min(0.12, duration * 0.0015))

#     age = patient.get("Age") or 45
#     add("Age", f"{age} years", -0.002 * max(0, age - 40))

#     probability = _sigmoid(logit)
#     factors.sort(key=lambda f: f["magnitude"], reverse=True)
#     return {"prediction": 1 if probability >= 0.5 else 0,
#             "probability": round(probability, 4), "factors": factors, "source": "offline"}


# # ===========================================================================
# # SECTION C: Flask API client  (database.db persistence layer)
# # ===========================================================================
# def api_ping(base_url):
#     try:
#         return requests.get(f"{base_url.rstrip('/')}/", timeout=2).status_code == 200
#     except requests.RequestException:
#         return False


# def api_save_patient(base_url, patient, result):
#     payload = {**patient, "prediction": result["prediction"],
#                "probability": result["probability"], "factors": result.get("factors", []),
#                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#     resp = requests.post(f"{base_url.rstrip('/')}/patients", json=payload, timeout=5)
#     resp.raise_for_status()
#     return resp.json()


# def api_get_patients(base_url):
#     resp = requests.get(f"{base_url.rstrip('/')}/patients", timeout=5)
#     resp.raise_for_status()
#     return resp.json()


# def api_delete_patient(base_url, patient_id):
#     requests.delete(f"{base_url.rstrip('/')}/patients/{patient_id}", timeout=5)


# def api_clear_patients(base_url):
#     requests.delete(f"{base_url.rstrip('/')}/patients", timeout=5)


# def generate_sample_patient(rng, index):
#     """Generate one varied demo patient for analytics/testing."""
#     age = rng.randint(18, 82)
#     gender = rng.choice(["Male", "Female"])
#     living_area = rng.choice(["Urban", "Urban", "Rural", "Semi-Urban"])

#     diabetes = 1 if rng.random() < (0.08 + max(0, age - 35) * 0.006) else 0
#     hypertension = 1 if rng.random() < (0.10 + max(0, age - 30) * 0.009) else 0
#     heart_disease = 1 if rng.random() < (0.04 + max(0, age - 40) * 0.004 + 0.05 * diabetes) else 0

#     systolic = rng.randint(105, 175) if hypertension else rng.randint(105, 135)
#     diastolic = rng.randint(65, 105) if hypertension else rng.randint(65, 88)
#     glucose = round(rng.uniform(80, 115) + (45 if diabetes else 0) + rng.uniform(-8, 15), 1)
#     cholesterol = round(rng.uniform(145, 225) + (25 if heart_disease else 0), 1)
#     bmi = round(rng.uniform(18.5, 34.5), 1)

#     past_options = [
#         "None", "Arthritis", "Kidney Disease", "Asthma",
#         "Thyroid Disorder", "Liver Disease", "Stroke History",
#     ]
#     past_disease = rng.choice(past_options)
#     if heart_disease and rng.random() < 0.45:
#         past_disease = "Heart Disease"
#     elif diabetes and rng.random() < 0.35:
#         past_disease = "Diabetes"

#     activity = round(rng.uniform(0.5, 7.0), 1)
#     sleep = round(rng.uniform(5.0, 9.0), 1)
#     diet = rng.choices(["Good", "Average", "Poor"], weights=[25, 50, 25], k=1)[0]
#     smoking = rng.choices(["Never", "Former", "Current"], weights=[55, 20, 25], k=1)[0]
#     stress = rng.choices(["Low", "Moderate", "High"], weights=[25, 50, 25], k=1)[0]
#     treatment = rng.choice(["Medication", "Lifestyle Changes", "Surgery", "Physiotherapy", "Combined Therapy"])
#     duration = rng.randint(7, 240)
#     follow_up = rng.choices(["Yes", "No"], weights=[78, 22], k=1)[0]

#     return {
#         **DEFAULT_PATIENT,
#         "Age": age, "Gender": gender, "Living_Area": living_area,
#         "Systolic_BP": systolic, "Diastolic_BP": diastolic,
#         "Glucose": glucose, "Cholesterol": cholesterol, "BMI": bmi,
#         "Diabetes": diabetes, "Hypertension": hypertension, "Heart_Disease": heart_disease,
#         "Past_Diseases": past_disease,
#         "Physical_Activity_Hours_Week": activity, "Sleep_Hours_Per_Night": sleep,
#         "Diet_Quality": diet, "Smoking_Status": smoking, "Stress_Level": stress,
#         "Treatment": treatment, "Treatment_Duration": duration, "Follow_Up": follow_up,
#     }


# def load_sample_dataset(base_url, count=100):
#     """Create and persist a reproducible demo dataset through the normal API."""
#     rng = random.Random(20260928)
#     saved = 0
#     for i in range(count):
#         patient = generate_sample_patient(rng, i + 1)
#         result = predict_offline(patient)
#         # Keep sample records distributed across recent dates for time-series charts.
#         payload_result = dict(result)
#         saved_record = api_save_patient(base_url, patient, payload_result)
#         saved += 1
#     return saved


# # ===========================================================================
# # SECTION D: Session state
# # ===========================================================================
# defaults = {
#     "patient": copy.deepcopy(DEFAULT_PATIENT), "result": None, "is_saved": False,
#     "active_persona_id": None, "api_url": API_BASE_URL_DEFAULT, "backend_online": False,
#     "form_section": "Vitals & Labs", "active_tab": "Dashboard",
# }
# for k, v in defaults.items():
#     st.session_state.setdefault(k, v)


# def reset_form():
#     st.session_state.patient = copy.deepcopy(DEFAULT_PATIENT)
#     st.session_state.result = None
#     st.session_state.is_saved = False
#     st.session_state.active_persona_id = None


# def apply_persona(persona):
#     st.session_state.patient = copy.deepcopy(persona["patient"])
#     st.session_state.active_persona_id = persona["id"]
#     st.session_state.result = None
#     st.session_state.is_saved = False


# def fetch_history_df():
#     """Pulls patient history from the Flask/SQLite backend into a DataFrame.
#     Returns (df, backend_reachable)."""
#     try:
#         records = api_get_patients(st.session_state.api_url)
#         st.session_state.backend_online = True
#         return pd.DataFrame(records), True
#     except requests.RequestException:
#         st.session_state.backend_online = False
#         return pd.DataFrame(), False


# # ===========================================================================
# # SECTION E: Left sidebar navigation + header
# # ===========================================================================
# NAV_ITEMS = [
#     ("Dashboard", "📊"),
#     ("New Prediction", "✨"),
#     ("Patient History", "📜"),
#     ("Disease Analysis", "🦠"),
#     ("Treatment Analysis", "💊"),
#     ("Risk Analysis", "⚠️"),
# ]


# def render_sidebar_nav():
#     with st.sidebar:
#         st.markdown("## 💓 Health Predictor")
#         # st.caption("`ML Model v1.0` — Clinical Decision Support")
#         st.divider()

#         labels = [f"{icon}  {name}" for name, icon in NAV_ITEMS]
#         names = [name for name, _ in NAV_ITEMS]
#         current_index = names.index(st.session_state.active_tab) if st.session_state.active_tab in names else 0

#         selected_label = st.radio(
#             "Navigation", labels, index=current_index, label_visibility="collapsed"
#         )
#         selected_name = names[labels.index(selected_label)]
#         st.session_state.active_tab = selected_name

#         st.divider()
#         st.markdown("### 🧪 Demo Dataset")
#         sample_count = st.selectbox("Sample records", [50, 75, 100], index=2)
#         if st.button("📥 Load Sample Data", use_container_width=True):
#             if not api_ping(st.session_state.api_url):
#                 st.error("Backend is not reachable. Start `flask_backend.py` first.")
#             else:
#                 progress = st.progress(0, text="Loading sample patients...")
#                 try:
#                     # Save in small batches so the UI remains responsive enough for a demo.
#                     rng = random.Random(20260928)
#                     for i in range(sample_count):
#                         patient = generate_sample_patient(rng, i + 1)
#                         result = predict_offline(patient)
#                         api_save_patient(st.session_state.api_url, patient, result)
#                         progress.progress((i + 1) / sample_count, text=f"Loading patient {i + 1}/{sample_count}")
#                     progress.empty()
#                     st.session_state.backend_online = True
#                     st.success(f"Loaded {sample_count} sample patient records.")
#                     st.rerun()
#                 except requests.RequestException as exc:
#                     progress.empty()
#                     st.error(f"Could not finish loading sample data: {exc}")

#         st.caption("Use this demo data to populate Dashboard and all analysis tabs.")
#         st.divider()
#         with st.expander("⚙️ Backend settings"):
#             st.session_state.api_url = st.text_input("Flask API base URL", value=st.session_state.api_url)
#             if st.button("🔄 Ping backend"):
#                 st.session_state.backend_online = api_ping(st.session_state.api_url)
#             st.caption("🟢 Online" if st.session_state.backend_online else "🟠 Not reachable yet")

#     return selected_name



# # ===========================================================================
# # SECTION F: Dashboard tab
# # ===========================================================================
# def render_dashboard():
#     st.subheader("📊 Dashboard")
#     df, online = fetch_history_df()
#     if not online:
#         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`, "
#                    "then click **Backend settings → Ping backend** above.")
#         return
#     if df.empty:
#         st.info("No assessments stored yet. Run a prediction in **New Prediction** and save it.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Assessments", len(df))
#     c2.metric("Improvement Rate", f"{round(df['prediction'].mean() * 100, 1)}%")
#     c3.metric("Average Age", f"{round(df['age'].mean(), 1)} yrs")
#     c4.metric("Average BMI", f"{round(df['bmi'].mean(), 1)}")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**Outcome Distribution**")
#         outcome_counts = df["prediction"].map({1: "Improved", 0: "Unimproved"}).value_counts()
#         st.bar_chart(outcome_counts)
#     with col2:
#         st.markdown("**Treatment Modality Distribution**")
#         st.bar_chart(df["treatment"].value_counts())

#     st.markdown("**Assessments Over Time**")
#     df["date"] = pd.to_datetime(df["timestamp"]).dt.date
#     st.line_chart(df.groupby("date").size().rename("Assessments"))


# # ===========================================================================
# # SECTION G: New Prediction tab (persona selector + form + result)
# # ===========================================================================
# def render_new_prediction():
#     st.markdown("##### Clinical Persona Presets  _(quick 1-click test cases)_")
#     cols = st.columns(len(PATIENT_PERSONAS))
#     for col, persona in zip(cols, PATIENT_PERSONAS):
#         with col:
#             is_active = st.session_state.active_persona_id == persona["id"]
#             if st.button(f"{'✅ ' if is_active else ''}**{persona['name']}**",
#                          key=f"persona_{persona['id']}", use_container_width=True):
#                 apply_persona(persona)
#                 st.rerun()
#             st.caption(f"`{persona['tag']}` — {persona['description']}")

#     if st.button("↻ Reset Form"):
#         reset_form()
#         st.rerun()

#     st.divider()
#     left, right = st.columns([3, 2])

#     with left:
#         p = st.session_state.patient
#         st.session_state.form_section = st.radio(
#             "Section", ["Vitals & Labs", "Medical History", "Lifestyle & Habits",
#                         "Treatment & Care", "Demographics"],
#             horizontal=True, label_visibility="collapsed",
#         )
#         section = st.session_state.form_section

#         if section == "Vitals & Labs":
#             bp = getBPCategory(p["Systolic_BP"], p["Diastolic_BP"])
#             st.markdown(f"{LEVEL_EMOJI[bp['level']]} **Blood Pressure status: {bp['label']}**")
#             c1, c2 = st.columns(2)
#             p["Systolic_BP"] = c1.slider("Systolic BP (mmHg)", 70, 200, p["Systolic_BP"])
#             p["Diastolic_BP"] = c2.slider("Diastolic BP (mmHg)", 40, 130, p["Diastolic_BP"])
#             c1, c2, c3 = st.columns(3)
#             with c1:
#                 p["Glucose"] = st.number_input("Blood Glucose (mg/dL)", 50, 300, p["Glucose"])
#                 st.caption(f"{LEVEL_EMOJI[getGlucoseCategory(p['Glucose'])['level']]} "
#                            f"{getGlucoseCategory(p['Glucose'])['label']}")
#             with c2:
#                 p["Cholesterol"] = st.number_input("Total Cholesterol (mg/dL)", 100, 400, p["Cholesterol"])
#                 st.caption(f"{LEVEL_EMOJI[getCholesterolCategory(p['Cholesterol'])['level']]} "
#                            f"{getCholesterolCategory(p['Cholesterol'])['label']}")
#             with c3:
#                 p["BMI"] = st.number_input("BMI (kg/m²)", 14.0, 50.0, float(p["BMI"]), step=0.1)
#                 st.caption(f"{LEVEL_EMOJI[getBMICategory(p['BMI'])['level']]} "
#                            f"{getBMICategory(p['BMI'])['label']}")

#         elif section == "Medical History":
#             c1, c2, c3 = st.columns(3)
#             p["Diabetes"] = int(c1.toggle("Diabetes Mellitus", value=bool(p["Diabetes"])))
#             p["Hypertension"] = int(c2.toggle("Hypertension", value=bool(p["Hypertension"])))
#             p["Heart_Disease"] = int(c3.toggle("Heart Disease", value=bool(p["Heart_Disease"])))
#             options = ["None", "Asthma", "Arthritis", "Kidney Disease", "Liver Disease", "Thyroid"]
#             p["Past_Diseases"] = st.radio("Primary Past Disease or Organ History", options,
#                                            index=options.index(p["Past_Diseases"]), horizontal=True)

#         elif section == "Lifestyle & Habits":
#             c1, c2 = st.columns(2)
#             p["Physical_Activity_Hours_Week"] = c1.slider("Physical Activity (hrs/week)", 0.0, 15.0,
#                                                            float(p["Physical_Activity_Hours_Week"]), step=0.5)
#             p["Sleep_Hours_Per_Night"] = c2.slider("Sleep Duration (hrs/night)", 3.0, 12.0,
#                                                    float(p["Sleep_Hours_Per_Night"]), step=0.5)
#             c1, c2, c3 = st.columns(3)
#             p["Diet_Quality"] = c1.radio("Diet Quality", ["Good", "Average", "Poor"],
#                                           index=["Good", "Average", "Poor"].index(p["Diet_Quality"]))
#             p["Smoking_Status"] = c2.radio("Smoking Status", ["Never", "Former", "Current"],
#                                             index=["Never", "Former", "Current"].index(p["Smoking_Status"]))
#             p["Stress_Level"] = c3.radio("Stress Level", ["Low", "Moderate", "High"],
#                                           index=["Low", "Moderate", "High"].index(p["Stress_Level"]))

#         elif section == "Treatment & Care":
#             c1, c2 = st.columns(2)
#             with c1:
#                 options = ["Medication", "Lifestyle Changes", "Surgery"]
#                 p["Treatment"] = st.radio("Treatment Modality", options, index=options.index(p["Treatment"]))
#             with c2:
#                 p["Treatment_Duration"] = st.slider("Treatment Duration (days)", 1, 365, p["Treatment_Duration"])
#                 p["Follow_Up"] = "Yes" if st.toggle("Attended Scheduled Follow-Up",
#                                                       value=(p["Follow_Up"] == "Yes")) else "No"

#         elif section == "Demographics":
#             c1, c2, c3 = st.columns(3)
#             p["Age"] = c1.slider("Age (years)", 18, 100, p["Age"])
#             p["Gender"] = c2.radio("Gender", ["Female", "Male"], index=["Female", "Male"].index(p["Gender"]))
#             options = ["Urban", "Suburban", "Rural"]
#             p["Living_Area"] = c3.radio("Living Area", options, index=options.index(p["Living_Area"]))

#         st.session_state.patient = p
#         st.divider()
#         if st.button("✨ Predict Patient Outcome", type="primary", use_container_width=True):
#             st.session_state.result = predict_offline(st.session_state.patient)
#             st.session_state.is_saved = False
#             st.rerun()

#     with right:
#         result = st.session_state.result
#         if not result:
#             st.info("⏳ Awaiting Patient Assessment — fill in the form, then click "
#                      "**Predict Patient Outcome**.")
#         else:
#             is_improved = result["prediction"] == 1
#             percent = round(result["probability"] * 1000) / 10
#             if is_improved:
#                 st.success("### ✅ Condition Likely to Improve")
#             else:
#                 st.error("### ⚠️ Guarded Prognosis / Low Improvement")
#             st.metric("Probability of Improvement", f"{percent}%")
#             st.progress(min(1.0, max(0.03, percent / 100)))

#             if st.session_state.is_saved:
#                 st.button("✅ Saved to Database", disabled=True, use_container_width=True)
#             else:
#                 if st.button("💾 Save to Database (database.db)", use_container_width=True):
#                     try:
#                         api_save_patient(st.session_state.api_url, st.session_state.patient, result)
#                         st.session_state.is_saved = True
#                         st.success("Saved to database.db via Flask backend.")
#                         st.rerun()
#                     except requests.RequestException as e:
#                         st.error(f"Could not reach Flask backend: {e}")

#     if st.session_state.result:
#         st.divider()
#         c1, c2 = st.columns(2)
#         with c1:
#             render_factor_impact(st.session_state.result)
#         with c2:
#             render_clinical_advice(st.session_state.patient, st.session_state.result)
#         render_report_section(st.session_state.patient, st.session_state.result)


# def render_factor_impact(result):
#     factors = result.get("factors", [])
#     if not factors:
#         return
#     st.markdown("##### 🔎 Explainable AI • Feature Contributions")
#     filter_choice = st.radio("Filter", ["All (top 8)", "Protective only", "Risk drivers only"],
#                               horizontal=True, label_visibility="collapsed", key="factor_filter")
#     if filter_choice == "Protective only":
#         shown = [f for f in factors if f["isPositive"]]
#     elif filter_choice == "Risk drivers only":
#         shown = [f for f in factors if not f["isPositive"]]
#     else:
#         shown = factors[:8]
#     df = pd.DataFrame([{"Factor": f["name"], "Impact": f["impact"]} for f in shown]).set_index("Factor")
#     st.bar_chart(df["Impact"])
#     for f in shown:
#         icon = "📈" if f["isPositive"] else "📉"
#         st.caption(f"{icon} **{f['name']}** ({f['description']}) — "
#                    f"{'+' if f['isPositive'] else ''}{f['impact']}")


# def render_clinical_advice(patient, result):
#     st.markdown("##### 🩺 Tailored Clinical Recommendations")
#     recs = generateClinicalRecommendations(patient, result["probability"])
#     badge = {"high": "🔴 High Priority", "medium": "🟠 Moderate Priority", "low": "🟢 Maintenance"}
#     for rec in recs:
#         with st.container(border=True):
#             st.markdown(f"**{rec['title']}**  —  {badge.get(rec['priority'], '')}")
#             st.caption(rec["detail"])


# def render_report_section(patient, result):
#     is_improved = result["prediction"] == 1
#     percent = round(result["probability"] * 1000) / 10
#     recs = generateClinicalRecommendations(patient, result["probability"])
#     with st.expander("🖨️ Full Clinical Report (printable summary)"):
#         lines = [
#             "# PulsePredict Clinical Assessment",
#             f"Date: {datetime.now().strftime('%Y-%m-%d')}",
#             f"## Model Prediction: {'Favorable' if is_improved else 'Guarded'} Prognosis",
#             f"Improvement Probability: **{percent}%**", "",
#             "## Patient Parameters",
#             f"- Demographics: {patient['Age']} yrs • {patient['Gender']} • {patient['Living_Area']}",
#             f"- Blood Pressure: {patient['Systolic_BP']}/{patient['Diastolic_BP']} mmHg",
#             f"- Glucose: {patient['Glucose']} mg/dL • Cholesterol: {patient['Cholesterol']} mg/dL • BMI: {patient['BMI']}",
#             f"- Treatment: {patient['Treatment']} ({patient['Treatment_Duration']} days) • Follow-Up: {patient['Follow_Up']}",
#             "", "## Recommendations",
#         ] + [f"- **{r['title']}**: {r['detail']}" for r in recs]
#         report_md = "\n".join(lines)
#         st.markdown(report_md)
#         st.download_button("⬇ Download Report (Markdown)", data=report_md,
#                            file_name=f"PulsePredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
#                            mime="text/markdown")


# # ===========================================================================
# # SECTION H: Patient History tab (reads database.db via Flask)
# # ===========================================================================
# def render_patient_history():
#     st.subheader("📜 Patient History")
#     df, online = fetch_history_df()
#     if not online:
#         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
#         return
#     if df.empty:
#         st.info("No assessments saved yet.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     gender_filter = c1.multiselect("Filter by gender", sorted(df["gender"].unique()))
#     treatment_filter = c2.multiselect("Filter by treatment", sorted(df["treatment"].unique()))
#     outcome_filter = c3.selectbox("Filter by outcome", ["All", "Improved", "Unimproved"])
#     sort_col = c4.selectbox("Sort by", ["id", "timestamp", "age", "probability"], index=0)

#     filtered = df.copy()
#     if gender_filter:
#         filtered = filtered[filtered["gender"].isin(gender_filter)]
#     if treatment_filter:
#         filtered = filtered[filtered["treatment"].isin(treatment_filter)]
#     if outcome_filter != "All":
#         filtered = filtered[filtered["prediction"] == (1 if outcome_filter == "Improved" else 0)]

#     # Ascending order (oldest / lowest first) as requested
#     filtered = filtered.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

#     st.dataframe(
#         filtered[["id", "timestamp", "age", "gender", "living_area", "systolic_bp", "diastolic_bp",
#                   "glucose", "treatment", "prediction", "probability"]],
#         use_container_width=True, hide_index=True,
#     )

#     st.caption("Select a record ID below to load it back into the New Prediction form, or delete it.")
#     c1, c2, c3 = st.columns(3)
#     selected_id = c1.number_input("Record ID", min_value=0, step=1, value=0)
#     if c2.button("↪ Load into form") and selected_id in filtered["id"].values:
#         row = filtered[filtered["id"] == selected_id].iloc[0]
#         st.session_state.patient = {
#             "Age": int(row["age"]), "Gender": row["gender"], "Living_Area": row["living_area"],
#             "Systolic_BP": int(row["systolic_bp"]), "Diastolic_BP": int(row["diastolic_bp"]),
#             "Glucose": row["glucose"], "Cholesterol": row["cholesterol"], "BMI": row["bmi"],
#             "Diabetes": int(row["diabetes"]), "Hypertension": int(row["hypertension"]),
#             "Heart_Disease": int(row["heart_disease"]), "Past_Diseases": row["past_diseases"],
#             "Physical_Activity_Hours_Week": row["physical_activity"], "Sleep_Hours_Per_Night": row["sleep_hours"],
#             "Diet_Quality": row["diet_quality"], "Smoking_Status": row["smoking_status"],
#             "Stress_Level": row["stress_level"], "Treatment": row["treatment"],
#             "Treatment_Duration": int(row["treatment_duration"]), "Follow_Up": row["follow_up"],
#         }
#         st.session_state.result = None
#         st.success("Loaded. Switch to the **New Prediction** tab to view/edit it.")
#     if c3.button("🗑 Delete record") and selected_id:
#         api_delete_patient(st.session_state.api_url, int(selected_id))
#         st.rerun()

#     if st.button("🗑 Clear ALL history", type="secondary"):
#         api_clear_patients(st.session_state.api_url)
#         st.rerun()


# # ===========================================================================
# # SECTION I: Disease Analysis tab
# # ===========================================================================
# def _make_disease_pie_chart(pie_df, label):
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(3.0, 2.6))
#     values = pie_df["Patients"].tolist()
#     labels = pie_df.index.tolist()
#     if sum(values) == 0:
#         values = [1, 0]
#     ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
#     ax.set_title(label, fontsize=11)
#     fig.tight_layout()
#     return fig


# def render_disease_analysis():
#     st.subheader("🦠 Disease Analysis")
#     st.caption("Disease-wise percentage, bar graph and pie chart from saved patient assessments.")

#     df, online = fetch_history_df()
#     if not online:
#         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
#         return
#     if df.empty:
#         st.info("No data yet — save some assessments first to generate disease analytics.")
#         return

#     disease_cols = [
#         ("diabetes", "Diabetes"),
#         ("hypertension", "Hypertension"),
#         ("heart_disease", "Heart Disease"),
#     ]
#     total = len(df)

#     for col, _ in disease_cols:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
#     df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)
#     df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)

#     st.markdown("### 📊 Disease-wise Analysis")

#     # Every disease is displayed as one horizontal row:
#     # Percentage | Bar Graph | Pie Chart
#     for col, label in disease_cols:
#         affected = int(df[col].sum())
#         absent = total - affected
#         prevalence = (affected / total * 100) if total else 0

#         affected_df = df[df[col] == 1]
#         improved_pct = affected_df["prediction"].mean() * 100 if not affected_df.empty else 0
#         avg_probability = affected_df["probability"].mean() * 100 if not affected_df.empty else 0

#         st.markdown(f"#### 🩺 {label}")
#         pct_col, bar_col, pie_col = st.columns([0.85, 1.7, 1.2])

#         with pct_col:
#             st.metric("Prevalence", f"{prevalence:.1f}%")
#             st.caption(f"{affected} of {total} patients")
#             st.metric("Improved", f"{improved_pct:.1f}%")
#             st.caption(f"Avg probability: {avg_probability:.1f}%")

#         with bar_col:
#             st.markdown("**Bar Graph — Present vs Absent**")
#             bar_df = pd.DataFrame({
#                 "Group": ["Present", "Absent"],
#                 "Patients": [affected, absent],
#             }).set_index("Group")
#             st.bar_chart(bar_df, y="Patients", height=240)

#         with pie_col:
#             st.markdown("**Pie Chart — Distribution**")
#             pie_df = pd.DataFrame({
#                 "Status": ["Present", "Absent"],
#                 "Patients": [affected, absent],
#             }).set_index("Status")
#             st.pyplot(_make_disease_pie_chart(pie_df, label), use_container_width=True)

#         st.divider()

#     # Compact comparison of all three diseases.
#     st.markdown("### 📈 Disease Percentage Comparison")
#     comparison = []
#     for col, label in disease_cols:
#         affected = int(df[col].sum())
#         affected_df = df[df[col] == 1]
#         comparison.append({
#             "Disease": label,
#             "Prevalence %": round((affected / total * 100), 1),
#             "Improved %": round(affected_df["prediction"].mean() * 100, 1) if not affected_df.empty else 0,
#             "Avg Probability %": round(affected_df["probability"].mean() * 100, 1) if not affected_df.empty else 0,
#         })

#     comparison_df = pd.DataFrame(comparison).set_index("Disease")
#     st.dataframe(
#         comparison_df.style.format("{:.1f}%"),
#         use_container_width=True,
#     )

#     st.markdown("### 📋 Past Disease / Organ History")
#     if "past_diseases" in df.columns:
#         past = df["past_diseases"].fillna("None").astype(str).str.strip()
#         past = past[past.str.lower().ne("none") & past.ne("")]

#         if past.empty:
#             st.info("No past disease / organ history has been recorded in saved assessments.")
#         else:
#             past_items = past.str.split(",").explode().astype(str).str.strip()
#             past_items = past_items[past_items.ne("")]
#             past_counts = past_items.value_counts().rename_axis("Past Disease").to_frame("Patients")
#             past_counts["Percentage"] = (past_counts["Patients"] / total * 100).round(1)
#             pc1, pc2 = st.columns([1.4, 1])
#             with pc1:
#                 st.markdown("**Frequency of recorded past diseases**")
#                 st.bar_chart(past_counts["Patients"], height=300)
#             with pc2:
#                 st.markdown("**Percentage**")
#                 st.dataframe(
#                     past_counts.assign(Percentage=past_counts["Percentage"].map(lambda x: f"{x:.1f}%")),
#                     use_container_width=True,
#                 )

#     st.caption("Percentages are calculated from the saved patient assessments currently available in the backend database. This is descriptive analysis and does not establish medical causation.")


# # ===========================================================================
# # SECTION J: Treatment Analysis tab
# # ===========================================================================
# def render_treatment_analysis():
#     st.subheader("💊 Treatment Analysis")
#     df, online = fetch_history_df()
#     if not online or df.empty:
#         st.info("No data yet — save some assessments first.")
#         return

#     st.markdown("**Average improvement probability by treatment modality**")
#     st.bar_chart(df.groupby("treatment")["probability"].mean())

#     st.markdown("**Improvement rate: Follow-up attended vs. missed**")
#     st.bar_chart(df.groupby("follow_up")["prediction"].mean())

#     st.markdown("**Treatment duration vs. improvement probability**")
#     st.scatter_chart(df, x="treatment_duration", y="probability", color="treatment")


# # ===========================================================================
# # SECTION K: Risk Analysis tab
# # ===========================================================================
# def render_risk_analysis():
#     st.subheader("⚠️ Risk Analysis")
#     df, online = fetch_history_df()
#     if not online or df.empty:
#         st.info("No data yet — save some assessments first.")
#         return

#     st.markdown("**Probability distribution across all assessments**")
#     st.bar_chart(pd.cut(df["probability"], bins=10).value_counts().sort_index().rename(
#         lambda i: str(i)))

#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("**Average probability by smoking status**")
#         st.bar_chart(df.groupby("smoking_status")["probability"].mean())
#     with c2:
#         st.markdown("**Average probability by stress level**")
#         st.bar_chart(df.groupby("stress_level")["probability"].mean())

#     st.markdown("**BMI category breakdown**")
#     bmi_cats = df["bmi"].apply(lambda b: getBMICategory(b)["label"])
#     st.bar_chart(bmi_cats.value_counts())


# # ===========================================================================
# # Page assembly
# # ===========================================================================
# active = render_sidebar_nav()

# st.markdown("## 💓 Patient Health Outcome Predictor")
# st.caption("Clinical Health Outcome Prognosis & Decision Support")
# st.divider()

# if active == "Dashboard":
#     render_dashboard()
# elif active == "New Prediction":
#     render_new_prediction()
# elif active == "Patient History":
#     render_patient_history()
# elif active == "Disease Analysis":
#     render_disease_analysis()
# elif active == "Treatment Analysis":
#     render_treatment_analysis()
# elif active == "Risk Analysis":
#     render_risk_analysis()
    
    
    
    
    
#     # """
# # streamlit_app.py
# # -----------------
# # PulsePredict - Patient Health Outcome Predictor (single-file Streamlit app)

# # Everything the previous 3-file version split across clinical_data.py /
# # predictor.py / app.py now lives in this one file, plus a new top navigation
# # bar with 6 sections:

# #     Dashboard | New Prediction | Patient History |
# #     Disease Analysis | Treatment Analysis | Risk Analysis

# # Persistence: every saved assessment is POSTed to a Flask backend
# # (flask_backend.py) which stores it in database.db (SQLite). Patient
# # History / Dashboard / the 3 analysis tabs all read from that database, so
# # your data survives across Streamlit restarts -- not just session memory.

# # Run (two terminals):
# #     Terminal 1:  python flask_backend.py        # starts DB + API on :5000
# #     Terminal 2:  streamlit run streamlit_app.py  # starts the UI
# # """

# # import copy
# # import json
# # import math
# # from datetime import datetime

# # import pandas as pd
# # import requests
# # import streamlit as st

# # st.set_page_config(
# #     page_title="PulsePredict - Patient Health Outcome Predictor",
# #     page_icon="💓",
# #     layout="wide",
# # )

# # API_BASE_URL_DEFAULT = "http://127.0.0.1:5000"


# # # ===========================================================================
# # # SECTION A: Clinical reference data  (formerly clinical_data.py)
# # # ===========================================================================
# # DEFAULT_PATIENT = {
# #     "Age": 45, "Gender": "Male", "Living_Area": "Urban",
# #     "Systolic_BP": 120, "Diastolic_BP": 80,
# #     "Glucose": 100, "Cholesterol": 180, "BMI": 24.5,
# #     "Diabetes": 0, "Hypertension": 0, "Heart_Disease": 0,
# #     "Past_Diseases": "None",
# #     "Physical_Activity_Hours_Week": 2.5, "Sleep_Hours_Per_Night": 7.0,
# #     "Diet_Quality": "Average", "Smoking_Status": "Never", "Stress_Level": "Moderate",
# #     "Treatment": "Medication", "Treatment_Duration": 30, "Follow_Up": "Yes",
# # }

# # LEVEL_EMOJI = {"good": "🟢", "watch": "🟡", "warn": "🟠", "bad": "🔴", "low": "🔵", "critical": "🟣"}


# # def getBPCategory(systolic, diastolic):
# #     systolic, diastolic = systolic or 0, diastolic or 0
# #     if systolic < 90 or diastolic < 60:
# #         return {"label": "Hypotension", "level": "low"}
# #     if systolic < 120 and diastolic < 80:
# #         return {"label": "Normal", "level": "good"}
# #     if systolic < 130 and diastolic < 80:
# #         return {"label": "Elevated", "level": "watch"}
# #     if systolic < 140 or diastolic < 90:
# #         return {"label": "Hypertension Stage 1", "level": "warn"}
# #     if systolic < 180 and diastolic < 120:
# #         return {"label": "Hypertension Stage 2", "level": "bad"}
# #     return {"label": "Hypertensive Crisis", "level": "critical"}


# # def getGlucoseCategory(glucose):
# #     glucose = glucose or 0
# #     if glucose < 70:
# #         return {"label": "Hypoglycemia", "level": "warn"}
# #     if glucose <= 99:
# #         return {"label": "Normal", "level": "good"}
# #     if glucose <= 125:
# #         return {"label": "Prediabetic Range", "level": "watch"}
# #     return {"label": "Diabetic Range", "level": "bad"}


# # def getCholesterolCategory(chol):
# #     chol = chol or 0
# #     if chol < 200:
# #         return {"label": "Desirable", "level": "good"}
# #     if chol < 240:
# #         return {"label": "Borderline High", "level": "watch"}
# #     return {"label": "High", "level": "bad"}


# # def getBMICategory(bmi):
# #     bmi = bmi or 0
# #     if bmi < 18.5:
# #         return {"label": "Underweight", "level": "watch"}
# #     if bmi < 25:
# #         return {"label": "Normal", "level": "good"}
# #     if bmi < 30:
# #         return {"label": "Overweight", "level": "watch"}
# #     return {"label": "Obese", "level": "bad"}


# # PATIENT_PERSONAS = [
# #     {"id": "optimal", "name": "Optimal Health Profile", "tag": "Low Risk",
# #      "description": "Healthy vitals, active lifestyle, full adherence.",
# #      "patient": {**DEFAULT_PATIENT, "Age": 34, "Systolic_BP": 115, "Diastolic_BP": 75,
# #                  "Glucose": 88, "Cholesterol": 170, "BMI": 22.0, "Physical_Activity_Hours_Week": 6.0,
# #                  "Sleep_Hours_Per_Night": 8.0, "Diet_Quality": "Good", "Stress_Level": "Low",
# #                  "Treatment": "Lifestyle Changes", "Treatment_Duration": 20}},
# #     {"id": "high-risk", "name": "High-Risk Comorbid Patient", "tag": "High Risk",
# #      "description": "Multiple chronic conditions, poor lifestyle indicators.",
# #      "patient": {**DEFAULT_PATIENT, "Age": 67, "Systolic_BP": 162, "Diastolic_BP": 98,
# #                  "Glucose": 185, "Cholesterol": 265, "BMI": 33.5, "Diabetes": 1, "Hypertension": 1,
# #                  "Heart_Disease": 1, "Past_Diseases": "Kidney Disease", "Physical_Activity_Hours_Week": 0.5,
# #                  "Sleep_Hours_Per_Night": 5.0, "Diet_Quality": "Poor", "Smoking_Status": "Current",
# #                  "Stress_Level": "High", "Treatment_Duration": 180, "Follow_Up": "No"}},
# #     {"id": "post-surgical", "name": "Post-Surgical Recovery", "tag": "Monitoring",
# #      "description": "Recent surgery, moderate vitals, good compliance.",
# #      "patient": {**DEFAULT_PATIENT, "Age": 52, "Systolic_BP": 128, "Diastolic_BP": 84,
# #                  "Glucose": 110, "Cholesterol": 205, "BMI": 26.0, "Hypertension": 1,
# #                  "Past_Diseases": "Arthritis", "Physical_Activity_Hours_Week": 1.5,
# #                  "Sleep_Hours_Per_Night": 6.5, "Smoking_Status": "Former",
# #                  "Treatment": "Surgery", "Treatment_Duration": 14}},
# #     {"id": "young-lifestyle", "name": "Young Adult, Lifestyle Risk", "tag": "Lifestyle Watch",
# #      "description": "Young patient with modifiable lifestyle risk factors.",
# #      "patient": {**DEFAULT_PATIENT, "Age": 26, "Systolic_BP": 124, "Diastolic_BP": 79,
# #                  "Glucose": 98, "Cholesterol": 195, "BMI": 27.5, "Physical_Activity_Hours_Week": 1.0,
# #                  "Sleep_Hours_Per_Night": 5.5, "Diet_Quality": "Poor", "Smoking_Status": "Current",
# #                  "Stress_Level": "High", "Treatment": "Lifestyle Changes", "Treatment_Duration": 10}},
# # ]


# # def generateClinicalRecommendations(patient, probability):
# #     recs = []
# #     bp = getBPCategory(patient.get("Systolic_BP"), patient.get("Diastolic_BP"))
# #     if bp["level"] in ("warn", "bad", "critical"):
# #         recs.append({"title": "Blood Pressure Management",
# #                       "detail": f"Classified as '{bp['label']}'. Recommend antihypertensive review and home BP monitoring.",
# #                       "priority": "high" if bp["level"] in ("bad", "critical") else "medium"})
# #     gl = getGlucoseCategory(patient.get("Glucose"))
# #     if gl["level"] in ("watch", "bad"):
# #         recs.append({"title": "Glycemic Control",
# #                       "detail": f"Glucose in '{gl['label']}' range. Suggest HbA1c testing and dietary review.",
# #                       "priority": "high" if gl["level"] == "bad" else "medium"})
# #     chol = getCholesterolCategory(patient.get("Cholesterol"))
# #     if chol["level"] in ("watch", "bad"):
# #         recs.append({"title": "Lipid Profile Follow-up",
# #                       "detail": f"Cholesterol is '{chol['label']}'. Consider statin evaluation and fasting lipid panel.",
# #                       "priority": "medium" if chol["level"] == "watch" else "high"})
# #     bmi = getBMICategory(patient.get("BMI"))
# #     if bmi["level"] in ("watch", "bad"):
# #         recs.append({"title": "Weight Management",
# #                       "detail": f"BMI is '{bmi['label']}'. Recommend nutrition counseling and activity ramp-up.",
# #                       "priority": "medium"})
# #     if patient.get("Smoking_Status") == "Current":
# #         recs.append({"title": "Smoking Cessation",
# #                       "detail": "Active smoking impairs recovery. Offer cessation counseling / NRT.",
# #                       "priority": "high"})
# #     if (patient.get("Physical_Activity_Hours_Week") or 0) < 2.5:
# #         recs.append({"title": "Increase Physical Activity",
# #                       "detail": "Below WHO-recommended 2.5 hrs/week. Suggest graded activity program.",
# #                       "priority": "medium"})
# #     if (patient.get("Sleep_Hours_Per_Night") or 0) < 6:
# #         recs.append({"title": "Sleep Hygiene",
# #                       "detail": "Sleep below 7-9h range may delay recovery. Consider sleep hygiene consult.",
# #                       "priority": "low"})
# #     if patient.get("Stress_Level") == "High":
# #         recs.append({"title": "Stress & Mental Health Support",
# #                       "detail": "High stress linked to poorer adherence. Recommend counseling referral.",
# #                       "priority": "medium"})
# #     if patient.get("Follow_Up") != "Yes":
# #         recs.append({"title": "Follow-Up Appointment Adherence",
# #                       "detail": "No follow-up attended -- strongest positive predictor. Reschedule immediately.",
# #                       "priority": "high"})
# #     if patient.get("Diabetes") or patient.get("Hypertension") or patient.get("Heart_Disease"):
# #         recs.append({"title": "Chronic Condition Co-Management",
# #                       "detail": "Comorbidities present. Recommend coordinated specialist care.",
# #                       "priority": "medium"})
# #     if not recs:
# #         recs.append({"title": "Maintain Current Regimen",
# #                       "detail": "All major biomarkers within healthy ranges. Continue routine check-ups.",
# #                       "priority": "low"})
# #     order = {"high": 0, "medium": 1, "low": 2}
# #     recs.sort(key=lambda r: order.get(r["priority"], 3))
# #     return recs


# # # ===========================================================================
# # # SECTION B: Offline prediction engine  (formerly predictor.py)
# # # ===========================================================================
# # def _sigmoid(x):
# #     return 1 / (1 + math.exp(-x))


# # def predict_offline(patient):
# #     """Transparent weighted-scoring stand-in for a trained ML model -- swap
# #     this out for model.predict_proba(...) once you have a real trained
# #     model.pkl; the rest of the app only cares about the returned dict shape."""
# #     factors = []
# #     logit = 0.0

# #     def add(name, description, weight):
# #         nonlocal logit
# #         logit += weight
# #         factors.append({"name": name, "description": description,
# #                          "impact": round(weight, 3), "magnitude": abs(weight),
# #                          "isPositive": weight > 0})

# #     add("Follow-Up Adherence",
# #         "Attended scheduled follow-up" if patient.get("Follow_Up") == "Yes" else "Missed follow-up appointment",
# #         0.275 if patient.get("Follow_Up") == "Yes" else -0.275)

# #     if patient.get("Diabetes"):
# #         add("Diabetes", "Diagnosed diabetes mellitus", -0.22)
# #     if patient.get("Hypertension"):
# #         add("Hypertension", "Diagnosed hypertension", -0.15)
# #     if patient.get("Heart_Disease"):
# #         add("Heart Disease", "Diagnosed cardiac history", -0.30)

# #     bmi = patient.get("BMI") or 24
# #     if bmi < 18.5 or bmi >= 30:
# #         add("BMI", f"BMI of {bmi} outside healthy range", -0.15)
# #     elif bmi >= 25:
# #         add("BMI", f"BMI of {bmi} slightly elevated", -0.07)
# #     else:
# #         add("BMI", f"BMI of {bmi} within healthy range", 0.08)

# #     glucose = patient.get("Glucose") or 100
# #     if glucose >= 126:
# #         add("Blood Glucose", f"{glucose} mg/dL - diabetic range", -0.18)
# #     elif glucose >= 100:
# #         add("Blood Glucose", f"{glucose} mg/dL - prediabetic range", -0.08)
# #     else:
# #         add("Blood Glucose", f"{glucose} mg/dL - normal range", 0.07)

# #     chol = patient.get("Cholesterol") or 180
# #     if chol >= 240:
# #         add("Cholesterol", f"{chol} mg/dL - high", -0.14)
# #     elif chol >= 200:
# #         add("Cholesterol", f"{chol} mg/dL - borderline high", -0.06)
# #     else:
# #         add("Cholesterol", f"{chol} mg/dL - desirable", 0.05)

# #     sys_bp, dia_bp = patient.get("Systolic_BP") or 120, patient.get("Diastolic_BP") or 80
# #     if sys_bp >= 140 or dia_bp >= 90:
# #         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - hypertensive", -0.16)
# #     elif sys_bp >= 130 or dia_bp >= 80:
# #         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - stage 1 range", -0.08)
# #     else:
# #         add("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg - normal range", 0.06)

# #     activity = patient.get("Physical_Activity_Hours_Week") or 0
# #     add("Physical Activity", f"{activity} hrs/week",
# #         min(0.15, activity * 0.03) if activity >= 2.5 else -0.05)

# #     sleep = patient.get("Sleep_Hours_Per_Night") or 7
# #     if 7 <= sleep <= 9:
# #         add("Sleep Duration", f"{sleep} hrs/night - optimal", 0.08)
# #     elif sleep < 6:
# #         add("Sleep Duration", f"{sleep} hrs/night - deficient", -0.10)
# #     else:
# #         add("Sleep Duration", f"{sleep} hrs/night - acceptable", 0.02)

# #     add("Diet Quality", patient.get("Diet_Quality", "Average"),
# #         {"Good": 0.10, "Average": 0.0, "Poor": -0.14}.get(patient.get("Diet_Quality"), 0.0))
# #     add("Smoking Status", patient.get("Smoking_Status", "Never"),
# #         {"Never": 0.08, "Former": -0.03, "Current": -0.22}.get(patient.get("Smoking_Status"), 0.0))
# #     add("Stress Level", patient.get("Stress_Level", "Moderate"),
# #         {"Low": 0.07, "Moderate": -0.02, "High": -0.16}.get(patient.get("Stress_Level"), 0.0))

# #     duration = patient.get("Treatment_Duration") or 30
# #     add("Treatment Duration", f"{duration} days of treatment", min(0.12, duration * 0.0015))

# #     age = patient.get("Age") or 45
# #     add("Age", f"{age} years", -0.002 * max(0, age - 40))

# #     probability = _sigmoid(logit)
# #     factors.sort(key=lambda f: f["magnitude"], reverse=True)
# #     return {"prediction": 1 if probability >= 0.5 else 0,
# #             "probability": round(probability, 4), "factors": factors, "source": "offline"}


# # # ===========================================================================
# # # SECTION C: Flask API client  (database.db persistence layer)
# # # ===========================================================================
# # def api_ping(base_url):
# #     try:
# #         return requests.get(f"{base_url.rstrip('/')}/", timeout=2).status_code == 200
# #     except requests.RequestException:
# #         return False


# # def api_save_patient(base_url, patient, result):
# #     payload = {**patient, "prediction": result["prediction"],
# #                "probability": result["probability"], "factors": result.get("factors", []),
# #                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# #     resp = requests.post(f"{base_url.rstrip('/')}/patients", json=payload, timeout=5)
# #     resp.raise_for_status()
# #     return resp.json()


# # def api_get_patients(base_url):
# #     resp = requests.get(f"{base_url.rstrip('/')}/patients", timeout=5)
# #     resp.raise_for_status()
# #     return resp.json()


# # def api_delete_patient(base_url, patient_id):
# #     requests.delete(f"{base_url.rstrip('/')}/patients/{patient_id}", timeout=5)


# # def api_clear_patients(base_url):
# #     requests.delete(f"{base_url.rstrip('/')}/patients", timeout=5)


# # # ===========================================================================
# # # SECTION D: Session state
# # # ===========================================================================
# # defaults = {
# #     "patient": copy.deepcopy(DEFAULT_PATIENT), "result": None, "is_saved": False,
# #     "active_persona_id": None, "api_url": API_BASE_URL_DEFAULT, "backend_online": False,
# #     "form_section": "Vitals & Labs", "active_tab": "Dashboard",
# # }
# # for k, v in defaults.items():
# #     st.session_state.setdefault(k, v)


# # def reset_form():
# #     st.session_state.patient = copy.deepcopy(DEFAULT_PATIENT)
# #     st.session_state.result = None
# #     st.session_state.is_saved = False
# #     st.session_state.active_persona_id = None


# # def apply_persona(persona):
# #     st.session_state.patient = copy.deepcopy(persona["patient"])
# #     st.session_state.active_persona_id = persona["id"]
# #     st.session_state.result = None
# #     st.session_state.is_saved = False


# # def fetch_history_df():
# #     """Pulls patient history from the Flask/SQLite backend into a DataFrame.
# #     Returns (df, backend_reachable)."""
# #     try:
# #         records = api_get_patients(st.session_state.api_url)
# #         st.session_state.backend_online = True
# #         return pd.DataFrame(records), True
# #     except requests.RequestException:
# #         st.session_state.backend_online = False
# #         return pd.DataFrame(), False


# # # ===========================================================================
# # # SECTION E: Top navigation bar + header
# # # ===========================================================================
# # def render_top_bar():
# #     c1, c2 = st.columns([3, 2])
# #     with c1:
# #         st.markdown("## 💓 PulsePredict  `ML Model v1.0`")
# #         st.caption("Clinical Health Outcome Prognosis & Decision Support")
# #     with c2:
# #         with st.popover("⚙️ Backend settings"):
# #             st.session_state.api_url = st.text_input("Flask API base URL", value=st.session_state.api_url)
# #             if st.button("🔄 Ping backend"):
# #                 st.session_state.backend_online = api_ping(st.session_state.api_url)
# #             st.caption("🟢 Online" if st.session_state.backend_online else "🟠 Not reachable yet")

# #     NAV_TABS = ["Dashboard", "New Prediction", "Patient History",
# #                 "Disease Analysis", "Treatment Analysis", "Risk Analysis"]
# #     tabs = st.tabs([f"  {t}  " for t in NAV_TABS])
# #     return dict(zip(NAV_TABS, tabs))


# # # ===========================================================================
# # # SECTION F: Dashboard tab
# # # ===========================================================================
# # def render_dashboard():
# #     st.subheader("📊 Dashboard")
# #     df, online = fetch_history_df()
# #     if not online:
# #         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`, "
# #                    "then click **Backend settings → Ping backend** above.")
# #         return
# #     if df.empty:
# #         st.info("No assessments stored yet. Run a prediction in **New Prediction** and save it.")
# #         return

# #     c1, c2, c3, c4 = st.columns(4)
# #     c1.metric("Total Assessments", len(df))
# #     c2.metric("Improvement Rate", f"{round(df['prediction'].mean() * 100, 1)}%")
# #     c3.metric("Average Age", f"{round(df['age'].mean(), 1)} yrs")
# #     c4.metric("Average BMI", f"{round(df['bmi'].mean(), 1)}")

# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown("**Outcome Distribution**")
# #         outcome_counts = df["prediction"].map({1: "Improved", 0: "Unimproved"}).value_counts()
# #         st.bar_chart(outcome_counts)
# #     with col2:
# #         st.markdown("**Treatment Modality Distribution**")
# #         st.bar_chart(df["treatment"].value_counts())

# #     st.markdown("**Assessments Over Time**")
# #     df["date"] = pd.to_datetime(df["timestamp"]).dt.date
# #     st.line_chart(df.groupby("date").size().rename("Assessments"))


# # # ===========================================================================
# # # SECTION G: New Prediction tab (persona selector + form + result)
# # # ===========================================================================
# # def render_new_prediction():
# #     st.markdown("##### Clinical Persona Presets  _(quick 1-click test cases)_")
# #     cols = st.columns(len(PATIENT_PERSONAS))
# #     for col, persona in zip(cols, PATIENT_PERSONAS):
# #         with col:
# #             is_active = st.session_state.active_persona_id == persona["id"]
# #             if st.button(f"{'✅ ' if is_active else ''}**{persona['name']}**",
# #                          key=f"persona_{persona['id']}", use_container_width=True):
# #                 apply_persona(persona)
# #                 st.rerun()
# #             st.caption(f"`{persona['tag']}` — {persona['description']}")

# #     if st.button("↻ Reset Form"):
# #         reset_form()
# #         st.rerun()

# #     st.divider()
# #     left, right = st.columns([3, 2])

# #     with left:
# #         p = st.session_state.patient
# #         st.session_state.form_section = st.radio(
# #             "Section", ["Vitals & Labs", "Medical History", "Lifestyle & Habits",
# #                         "Treatment & Care", "Demographics"],
# #             horizontal=True, label_visibility="collapsed",
# #         )
# #         section = st.session_state.form_section

# #         if section == "Vitals & Labs":
# #             bp = getBPCategory(p["Systolic_BP"], p["Diastolic_BP"])
# #             st.markdown(f"{LEVEL_EMOJI[bp['level']]} **Blood Pressure status: {bp['label']}**")
# #             c1, c2 = st.columns(2)
# #             p["Systolic_BP"] = c1.slider("Systolic BP (mmHg)", 70, 200, p["Systolic_BP"])
# #             p["Diastolic_BP"] = c2.slider("Diastolic BP (mmHg)", 40, 130, p["Diastolic_BP"])
# #             c1, c2, c3 = st.columns(3)
# #             with c1:
# #                 p["Glucose"] = st.number_input("Blood Glucose (mg/dL)", 50, 300, p["Glucose"])
# #                 st.caption(f"{LEVEL_EMOJI[getGlucoseCategory(p['Glucose'])['level']]} "
# #                            f"{getGlucoseCategory(p['Glucose'])['label']}")
# #             with c2:
# #                 p["Cholesterol"] = st.number_input("Total Cholesterol (mg/dL)", 100, 400, p["Cholesterol"])
# #                 st.caption(f"{LEVEL_EMOJI[getCholesterolCategory(p['Cholesterol'])['level']]} "
# #                            f"{getCholesterolCategory(p['Cholesterol'])['label']}")
# #             with c3:
# #                 p["BMI"] = st.number_input("BMI (kg/m²)", 14.0, 50.0, float(p["BMI"]), step=0.1)
# #                 st.caption(f"{LEVEL_EMOJI[getBMICategory(p['BMI'])['level']]} "
# #                            f"{getBMICategory(p['BMI'])['label']}")

# #         elif section == "Medical History":
# #             c1, c2, c3 = st.columns(3)
# #             p["Diabetes"] = int(c1.toggle("Diabetes Mellitus", value=bool(p["Diabetes"])))
# #             p["Hypertension"] = int(c2.toggle("Hypertension", value=bool(p["Hypertension"])))
# #             p["Heart_Disease"] = int(c3.toggle("Heart Disease", value=bool(p["Heart_Disease"])))
# #             options = ["None", "Asthma", "Arthritis", "Kidney Disease", "Liver Disease", "Thyroid"]
# #             p["Past_Diseases"] = st.radio("Primary Past Disease or Organ History", options,
# #                                            index=options.index(p["Past_Diseases"]), horizontal=True)

# #         elif section == "Lifestyle & Habits":
# #             c1, c2 = st.columns(2)
# #             p["Physical_Activity_Hours_Week"] = c1.slider("Physical Activity (hrs/week)", 0.0, 15.0,
# #                                                            float(p["Physical_Activity_Hours_Week"]), step=0.5)
# #             p["Sleep_Hours_Per_Night"] = c2.slider("Sleep Duration (hrs/night)", 3.0, 12.0,
# #                                                    float(p["Sleep_Hours_Per_Night"]), step=0.5)
# #             c1, c2, c3 = st.columns(3)
# #             p["Diet_Quality"] = c1.radio("Diet Quality", ["Good", "Average", "Poor"],
# #                                           index=["Good", "Average", "Poor"].index(p["Diet_Quality"]))
# #             p["Smoking_Status"] = c2.radio("Smoking Status", ["Never", "Former", "Current"],
# #                                             index=["Never", "Former", "Current"].index(p["Smoking_Status"]))
# #             p["Stress_Level"] = c3.radio("Stress Level", ["Low", "Moderate", "High"],
# #                                           index=["Low", "Moderate", "High"].index(p["Stress_Level"]))

# #         elif section == "Treatment & Care":
# #             c1, c2 = st.columns(2)
# #             with c1:
# #                 options = ["Medication", "Lifestyle Changes", "Surgery"]
# #                 p["Treatment"] = st.radio("Treatment Modality", options, index=options.index(p["Treatment"]))
# #             with c2:
# #                 p["Treatment_Duration"] = st.slider("Treatment Duration (days)", 1, 365, p["Treatment_Duration"])
# #                 p["Follow_Up"] = "Yes" if st.toggle("Attended Scheduled Follow-Up",
# #                                                       value=(p["Follow_Up"] == "Yes")) else "No"

# #         elif section == "Demographics":
# #             c1, c2, c3 = st.columns(3)
# #             p["Age"] = c1.slider("Age (years)", 18, 100, p["Age"])
# #             p["Gender"] = c2.radio("Gender", ["Female", "Male"], index=["Female", "Male"].index(p["Gender"]))
# #             options = ["Urban", "Suburban", "Rural"]
# #             p["Living_Area"] = c3.radio("Living Area", options, index=options.index(p["Living_Area"]))

# #         st.session_state.patient = p
# #         st.divider()
# #         if st.button("✨ Predict Patient Outcome", type="primary", use_container_width=True):
# #             st.session_state.result = predict_offline(st.session_state.patient)
# #             st.session_state.is_saved = False
# #             st.rerun()

# #     with right:
# #         result = st.session_state.result
# #         if not result:
# #             st.info("⏳ Awaiting Patient Assessment — fill in the form, then click "
# #                      "**Predict Patient Outcome**.")
# #         else:
# #             is_improved = result["prediction"] == 1
# #             percent = round(result["probability"] * 1000) / 10
# #             if is_improved:
# #                 st.success("### ✅ Condition Likely to Improve")
# #             else:
# #                 st.error("### ⚠️ Guarded Prognosis / Low Improvement")
# #             st.metric("Probability of Improvement", f"{percent}%")
# #             st.progress(min(1.0, max(0.03, percent / 100)))

# #             if st.session_state.is_saved:
# #                 st.button("✅ Saved to Database", disabled=True, use_container_width=True)
# #             else:
# #                 if st.button("💾 Save to Database (database.db)", use_container_width=True):
# #                     try:
# #                         api_save_patient(st.session_state.api_url, st.session_state.patient, result)
# #                         st.session_state.is_saved = True
# #                         st.success("Saved to database.db via Flask backend.")
# #                         st.rerun()
# #                     except requests.RequestException as e:
# #                         st.error(f"Could not reach Flask backend: {e}")

# #     if st.session_state.result:
# #         st.divider()
# #         c1, c2 = st.columns(2)
# #         with c1:
# #             render_factor_impact(st.session_state.result)
# #         with c2:
# #             render_clinical_advice(st.session_state.patient, st.session_state.result)
# #         render_report_section(st.session_state.patient, st.session_state.result)


# # def render_factor_impact(result):
# #     factors = result.get("factors", [])
# #     if not factors:
# #         return
# #     st.markdown("##### 🔎 Explainable AI • Feature Contributions")
# #     filter_choice = st.radio("Filter", ["All (top 8)", "Protective only", "Risk drivers only"],
# #                               horizontal=True, label_visibility="collapsed", key="factor_filter")
# #     if filter_choice == "Protective only":
# #         shown = [f for f in factors if f["isPositive"]]
# #     elif filter_choice == "Risk drivers only":
# #         shown = [f for f in factors if not f["isPositive"]]
# #     else:
# #         shown = factors[:8]
# #     df = pd.DataFrame([{"Factor": f["name"], "Impact": f["impact"]} for f in shown]).set_index("Factor")
# #     st.bar_chart(df["Impact"])
# #     for f in shown:
# #         icon = "📈" if f["isPositive"] else "📉"
# #         st.caption(f"{icon} **{f['name']}** ({f['description']}) — "
# #                    f"{'+' if f['isPositive'] else ''}{f['impact']}")


# # def render_clinical_advice(patient, result):
# #     st.markdown("##### 🩺 Tailored Clinical Recommendations")
# #     recs = generateClinicalRecommendations(patient, result["probability"])
# #     badge = {"high": "🔴 High Priority", "medium": "🟠 Moderate Priority", "low": "🟢 Maintenance"}
# #     for rec in recs:
# #         with st.container(border=True):
# #             st.markdown(f"**{rec['title']}**  —  {badge.get(rec['priority'], '')}")
# #             st.caption(rec["detail"])


# # def render_report_section(patient, result):
# #     is_improved = result["prediction"] == 1
# #     percent = round(result["probability"] * 1000) / 10
# #     recs = generateClinicalRecommendations(patient, result["probability"])
# #     with st.expander("🖨️ Full Clinical Report (printable summary)"):
# #         lines = [
# #             "# PulsePredict Clinical Assessment",
# #             f"Date: {datetime.now().strftime('%Y-%m-%d')}",
# #             f"## Model Prediction: {'Favorable' if is_improved else 'Guarded'} Prognosis",
# #             f"Improvement Probability: **{percent}%**", "",
# #             "## Patient Parameters",
# #             f"- Demographics: {patient['Age']} yrs • {patient['Gender']} • {patient['Living_Area']}",
# #             f"- Blood Pressure: {patient['Systolic_BP']}/{patient['Diastolic_BP']} mmHg",
# #             f"- Glucose: {patient['Glucose']} mg/dL • Cholesterol: {patient['Cholesterol']} mg/dL • BMI: {patient['BMI']}",
# #             f"- Treatment: {patient['Treatment']} ({patient['Treatment_Duration']} days) • Follow-Up: {patient['Follow_Up']}",
# #             "", "## Recommendations",
# #         ] + [f"- **{r['title']}**: {r['detail']}" for r in recs]
# #         report_md = "\n".join(lines)
# #         st.markdown(report_md)
# #         st.download_button("⬇ Download Report (Markdown)", data=report_md,
# #                            file_name=f"PulsePredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
# #                            mime="text/markdown")


# # # ===========================================================================
# # # SECTION H: Patient History tab (reads database.db via Flask)
# # # ===========================================================================
# # def render_patient_history():
# #     st.subheader("📜 Patient History")
# #     df, online = fetch_history_df()
# #     if not online:
# #         st.warning("Flask backend is not reachable. Start it with `python flask_backend.py`.")
# #         return
# #     if df.empty:
# #         st.info("No assessments saved yet.")
# #         return

# #     c1, c2, c3 = st.columns(3)
# #     gender_filter = c1.multiselect("Filter by gender", sorted(df["gender"].unique()))
# #     treatment_filter = c2.multiselect("Filter by treatment", sorted(df["treatment"].unique()))
# #     outcome_filter = c3.selectbox("Filter by outcome", ["All", "Improved", "Unimproved"])

# #     filtered = df.copy()
# #     if gender_filter:
# #         filtered = filtered[filtered["gender"].isin(gender_filter)]
# #     if treatment_filter:
# #         filtered = filtered[filtered["treatment"].isin(treatment_filter)]
# #     if outcome_filter != "All":
# #         filtered = filtered[filtered["prediction"] == (1 if outcome_filter == "Improved" else 0)]

# #     st.dataframe(
# #         filtered[["id", "timestamp", "age", "gender", "living_area", "systolic_bp", "diastolic_bp",
# #                   "glucose", "treatment", "prediction", "probability"]],
# #         use_container_width=True, hide_index=True,
# #     )

# #     st.caption("Select a record ID below to load it back into the New Prediction form, or delete it.")
# #     c1, c2, c3 = st.columns(3)
# #     selected_id = c1.number_input("Record ID", min_value=0, step=1, value=0)
# #     if c2.button("↪ Load into form") and selected_id in filtered["id"].values:
# #         row = filtered[filtered["id"] == selected_id].iloc[0]
# #         st.session_state.patient = {
# #             "Age": int(row["age"]), "Gender": row["gender"], "Living_Area": row["living_area"],
# #             "Systolic_BP": int(row["systolic_bp"]), "Diastolic_BP": int(row["diastolic_bp"]),
# #             "Glucose": row["glucose"], "Cholesterol": row["cholesterol"], "BMI": row["bmi"],
# #             "Diabetes": int(row["diabetes"]), "Hypertension": int(row["hypertension"]),
# #             "Heart_Disease": int(row["heart_disease"]), "Past_Diseases": row["past_diseases"],
# #             "Physical_Activity_Hours_Week": row["physical_activity"], "Sleep_Hours_Per_Night": row["sleep_hours"],
# #             "Diet_Quality": row["diet_quality"], "Smoking_Status": row["smoking_status"],
# #             "Stress_Level": row["stress_level"], "Treatment": row["treatment"],
# #             "Treatment_Duration": int(row["treatment_duration"]), "Follow_Up": row["follow_up"],
# #         }
# #         st.session_state.result = None
# #         st.success("Loaded. Switch to the **New Prediction** tab to view/edit it.")
# #     if c3.button("🗑 Delete record") and selected_id:
# #         api_delete_patient(st.session_state.api_url, int(selected_id))
# #         st.rerun()

# #     if st.button("🗑 Clear ALL history", type="secondary"):
# #         api_clear_patients(st.session_state.api_url)
# #         st.rerun()


# # # ===========================================================================
# # # SECTION I: Disease Analysis tab
# # # ===========================================================================
# # def render_disease_analysis():
# #     st.subheader("🦠 Disease Analysis")
# #     df, online = fetch_history_df()
# #     if not online or df.empty:
# #         st.info("No data yet — save some assessments first.")
# #         return

# #     for col, label in [("diabetes", "Diabetes"), ("hypertension", "Hypertension"), ("heart_disease", "Heart Disease")]:
# #         st.markdown(f"**Improvement rate by {label} status**")
# #         grouped = df.groupby(col)["prediction"].mean().rename(index={0: "No", 1: "Yes"})
# #         st.bar_chart(grouped)

# #     st.markdown("**Most common past disease / organ history**")
# #     st.bar_chart(df["past_diseases"].value_counts())


# # # ===========================================================================
# # # SECTION J: Treatment Analysis tab
# # # ===========================================================================
# # def render_treatment_analysis():
# #     st.subheader("💊 Treatment Analysis")
# #     df, online = fetch_history_df()
# #     if not online or df.empty:
# #         st.info("No data yet — save some assessments first.")
# #         return

# #     st.markdown("**Average improvement probability by treatment modality**")
# #     st.bar_chart(df.groupby("treatment")["probability"].mean())

# #     st.markdown("**Improvement rate: Follow-up attended vs. missed**")
# #     st.bar_chart(df.groupby("follow_up")["prediction"].mean())

# #     st.markdown("**Treatment duration vs. improvement probability**")
# #     st.scatter_chart(df, x="treatment_duration", y="probability", color="treatment")


# # # ===========================================================================
# # # SECTION K: Risk Analysis tab
# # # ===========================================================================
# # def render_risk_analysis():
# #     st.subheader("⚠️ Risk Analysis")
# #     df, online = fetch_history_df()
# #     if not online or df.empty:
# #         st.info("No data yet — save some assessments first.")
# #         return

# #     st.markdown("**Probability distribution across all assessments**")
# #     st.bar_chart(pd.cut(df["probability"], bins=10).value_counts().sort_index().rename(
# #         lambda i: str(i)))

# #     c1, c2 = st.columns(2)
# #     with c1:
# #         st.markdown("**Average probability by smoking status**")
# #         st.bar_chart(df.groupby("smoking_status")["probability"].mean())
# #     with c2:
# #         st.markdown("**Average probability by stress level**")
# #         st.bar_chart(df.groupby("stress_level")["probability"].mean())

# #     st.markdown("**BMI category breakdown**")
# #     bmi_cats = df["bmi"].apply(lambda b: getBMICategory(b)["label"])
# #     st.bar_chart(bmi_cats.value_counts())


# # # ===========================================================================
# # # Page assembly
# # # ===========================================================================
# # tab_map = render_top_bar()

# # with tab_map["Dashboard"]:
# #     render_dashboard()
# # with tab_map["New Prediction"]:
# #     render_new_prediction()
# # with tab_map["Patient History"]:
# #     render_patient_history()
# # with tab_map["Disease Analysis"]:
# #     render_disease_analysis()
# # with tab_map["Treatment Analysis"]:
# #     render_treatment_analysis()
# # with tab_map["Risk Analysis"]:
# #     render_risk_analysis()
