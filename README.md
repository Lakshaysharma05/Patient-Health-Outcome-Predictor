# PulsePredict &bull; Patient Health Outcome Predictor

A modern clinical decision support web application powered by Machine Learning to predict patient health outcome trajectories based on vital signs, comorbidities, lifestyle factors, and treatment adherence.

---

## 🌟 Key Features

1. **Dual Operation Modes**:
   - **Full-Stack Integrated Mode**: React communicates with the Flask REST API (`POST /predict` and `GET /health`).
   - **Offline Standalone Mode**: Embedded client-side mathematical model engine calculates exact predictions matching the Python trained model to 16 decimal places.
2. **Comprehensive Clinical Parameters (20 inputs, 37 one-hot features)**:
   - **Vitals & Labs**: Blood pressure sliders (Systolic & Diastolic) with real-time AHA/ACC classification, Fasting Glucose (ADA standards), Total Cholesterol, Body Mass Index (BMI).
   - **Medical History**: Comorbidity toggles for Diabetes, Hypertension, Heart Disease, and Past Major Diseases.
   - **Lifestyle & Habits**: Physical Activity (WHO guidelines), Sleep Duration, Diet Quality, Smoking Status, Stress Level.
   - **Treatment & Care**: Primary Modality (Medication, Lifestyle Changes, Surgery), Treatment Duration (1-365 days), Follow-Up Attendance.
3. **Clinical Decision Support & Explainable AI (XAI)**:
   - **Outcome Verdict**: Clear distinction between Favorable Prognosis vs. Guarded / High-Risk Prognosis with percentage odds.
   - **Feature Contribution Breakdown**: Protective factors (+) vs. Risk drivers (&minus;) based on trained weights and standardization.
   - **Tailored Recommendations**: Dynamic medical advice prioritized by urgency (Care Adherence, Blood Pressure control, Smoking Cessation, Sleep Hygiene).
4. **Interactive Workflow**:
   - **1-Click Persona Presets**: Instantly load realistic clinical cases (Optimal Recovery, High-Risk Cardiac & Metabolic, Post-Surgical Senior, Lifestyle Intervention).
   - **Randomize Patient**: Generate realistic randomized profiles for testing.
   - **Assessment History**: Save, compare, and reload previous evaluations.
   - **Printable Clinical Report**: PDF/print-ready summary with patient biomarker tables, model verdict, and clinician signature block.

---

## 🚀 Quick Start Guide

### Option 1: Run Full-Stack with Flask (Single Port :5000)

The React frontend is already built into `frontend/dist/`. Flask is configured to serve both the API and the React web app:

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run Flask backend
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

### Option 2: Run with Vite Dev Server (Hot Module Reloading)

If you are developing or modifying the React UI:

```bash
# In one terminal, run Flask backend:
python app.py

# In a second terminal, start the React dev server:
cd frontend
npm run dev
```

Open your browser and navigate to:
👉 **`http://localhost:5173`** (Vite automatically proxies `/predict` and `/health` to Flask)

---

## 📂 Project Structure

```
├── app.py                      # Flask REST backend & static file server
├── main.ipynb                  # Model training and EDA notebook
├── model/                      # Trained model artifacts
│   ├── weights.npy             # Logistic regression feature weights
│   ├── bias.npy                # Model intercept bias
│   ├── mean.npy                # StandardScaler feature means
│   ├── std.npy                 # StandardScaler feature standard deviations
│   └── feature_columns.json    # Exactly ordered 37 encoded columns
└── frontend/                   # React + Vite + Tailwind CSS application
    ├── src/
    │   ├── App.jsx             # Main interactive dashboard container
    │   ├── components/
    │   │   ├── Header.jsx          # Navigation, status indicator, action buttons
    │   │   ├── PersonaSelector.jsx # 1-click clinical test cases
    │   │   ├── PatientForm.jsx     # Tabbed clinical input form with live validation
    │   │   ├── OutcomeResult.jsx   # Verdict banner, probability gauge, decision tier
    │   │   ├── FactorImpact.jsx    # Explainable AI feature contributions
    │   │   ├── ClinicalAdvice.jsx  # Prioritized tailored clinical guidance
    │   │   ├── HistoryDrawer.jsx   # Saved assessment comparison history
    │   │   └── ReportModal.jsx     # Printable clinical report
    │   ├── data/
    │   │   ├── clinicalNorms.js    # AHA/ADA reference ranges & recommendation engine
    │   │   └── personas.js         # Preset patient profiles
    │   ├── services/
    │   │   └── predictor.js        # API caller & mathematically exact local predictor
    │   ├── modelParameters.json    # Exported model weights for offline fallback
    │   └── index.css               # Tailwind CSS styling & print formatting
    ├── package.json
    └── vite.config.js              # Vite configuration with Tailwind & Flask proxy
```
