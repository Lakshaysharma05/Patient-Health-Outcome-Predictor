"""
flask_backend.py
-----------------
Flask backend for PulsePredict. Creates and manages database.db (SQLite)
and exposes a small REST API that streamlit_app.py talks to for all
persistent storage (Patient History / Dashboard / Disease-Treatment-Risk
Analysis tabs all read from this database).

Run with:   python flask_backend.py
Default:    http://127.0.0.1:5000

Endpoints
---------
GET    /                 -> health check (used by the "ping backend" button)
POST   /patients          -> insert one patient assessment record
GET    /patients           -> list all records (newest first)
GET    /patients/<id>        -> get a single record
DELETE /patients/<id>          -> delete a single record
DELETE /patients                 -> clear all records
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, g, jsonify, request

DB_PATH = Path(__file__).parent / "database.db"

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            gender TEXT,
            living_area TEXT,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            glucose REAL,
            cholesterol REAL,
            bmi REAL,
            diabetes INTEGER,
            hypertension INTEGER,
            heart_disease INTEGER,
            past_diseases TEXT,
            physical_activity REAL,
            sleep_hours REAL,
            diet_quality TEXT,
            smoking_status TEXT,
            stress_level TEXT,
            treatment TEXT,
            treatment_duration INTEGER,
            follow_up TEXT,
            prediction INTEGER,
            probability REAL,
            factors_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def row_to_dict(row):
    d = dict(row)
    # Expand factors_json back into a Python list for the frontend
    d["factors"] = json.loads(d.pop("factors_json") or "[]")
    return d


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "PulsePredict Flask Backend"})


@app.route("/patients", methods=["POST"])
def create_patient():
    data = request.get_json(force=True)

    required = ["Age", "Gender", "Systolic_BP", "Diastolic_BP"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing mandatory fields: {missing}"}), 400

    db = get_db()
    cur = db.execute(
        """INSERT INTO patients (
            timestamp, age, gender, living_area, systolic_bp, diastolic_bp,
            glucose, cholesterol, bmi, diabetes, hypertension, heart_disease,
            past_diseases, physical_activity, sleep_hours, diet_quality,
            smoking_status, stress_level, treatment, treatment_duration,
            follow_up, prediction, probability, factors_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data.get("Age"), data.get("Gender"), data.get("Living_Area"),
            data.get("Systolic_BP"), data.get("Diastolic_BP"),
            data.get("Glucose"), data.get("Cholesterol"), data.get("BMI"),
            int(bool(data.get("Diabetes"))), int(bool(data.get("Hypertension"))),
            int(bool(data.get("Heart_Disease"))), data.get("Past_Diseases"),
            data.get("Physical_Activity_Hours_Week"), data.get("Sleep_Hours_Per_Night"),
            data.get("Diet_Quality"), data.get("Smoking_Status"), data.get("Stress_Level"),
            data.get("Treatment"), data.get("Treatment_Duration"), data.get("Follow_Up"),
            data.get("prediction"), data.get("probability"),
            json.dumps(data.get("factors", [])),
        ),
    )
    db.commit()
    new_id = cur.lastrowid
    row = db.execute("SELECT * FROM patients WHERE id = ?", (new_id,)).fetchone()
    return jsonify(row_to_dict(row)), 201


@app.route("/patients", methods=["GET"])
def list_patients():
    db = get_db()
    rows = db.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    return jsonify([row_to_dict(r) for r in rows])


@app.route("/patients/<int:patient_id>", methods=["GET"])
def get_patient(patient_id):
    db = get_db()
    row = db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row_to_dict(row))


@app.route("/patients/<int:patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    db = get_db()
    db.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    db.commit()
    return jsonify({"deleted": patient_id})


@app.route("/patients", methods=["DELETE"])
def clear_patients():
    db = get_db()
    db.execute("DELETE FROM patients")
    db.commit()
    return jsonify({"cleared": True})


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH.resolve()}")
    app.run(host="127.0.0.1", port=5000, debug=True)
