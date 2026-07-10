"""
Heart Disease Detection System - Flask Backend
------------------------------------------------
REST API that:
  - Loads the trained model (heart_model.pkl) and scaler (scaler.pkl)
  - Accepts patient data from the frontend and returns a JSON prediction
  - Stores every prediction in a local SQLite database
  - Exposes endpoints for history, search and dashboard statistics
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, g, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "predictions.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load ML artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_columns"]

FIELD_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (60, 250),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER, sex INTEGER, cp INTEGER, trestbps REAL, chol REAL,
            fbs INTEGER, restecg INTEGER, thalach REAL, exang INTEGER,
            oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER,
            prediction INTEGER, confidence REAL, risk_label TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_payload(data):
    errors = []
    values = {}
    for field in FEATURE_ORDER:
        if field not in data or data[field] in (None, ""):
            errors.append(f"'{field}' is required.")
            continue
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be numeric.")
            continue
        lo, hi = FIELD_RANGES[field]
        if not (lo <= val <= hi):
            errors.append(f"'{field}' must be between {lo} and {hi}.")
            continue
        values[field] = val
    return values, errors


def get_recommendation(risk_label, confidence):
    if risk_label == "High Risk":
        return (
            "Your results indicate elevated risk factors for heart disease. "
            "Please consult a cardiologist soon for a full clinical evaluation. "
            "In the meantime, monitor your blood pressure, reduce salt and "
            "saturated-fat intake, avoid smoking, and stay physically active "
            "within limits recommended by your doctor."
        )
    return (
        "Your results indicate a low likelihood of heart disease based on the "
        "provided values. Keep maintaining a heart-healthy lifestyle: regular "
        "exercise, a balanced diet, routine checkups, and stress management."
    )


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes - API
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    patient_name = data.get("patient_name", "Unnamed Patient")

    values, errors = validate_payload(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    ordered = [values[f] for f in FEATURE_ORDER]
    X_df = pd.DataFrame([ordered], columns=FEATURE_ORDER)
    X_scaled = scaler.transform(X_df)

    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 2)
    risk_label = "High Risk" if pred == 1 else "Low Risk"
    recommendation = get_recommendation(risk_label, confidence)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute(
        """INSERT INTO predictions
           (patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal, prediction, confidence, risk_label, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            patient_name, values["age"], values["sex"], values["cp"], values["trestbps"],
            values["chol"], values["fbs"], values["restecg"], values["thalach"],
            values["exang"], values["oldpeak"], values["slope"], values["ca"], values["thal"],
            pred, confidence, risk_label, timestamp,
        ),
    )
    db.commit()
    new_id = db.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

    return jsonify({
        "success": True,
        "id": new_id,
        "prediction": pred,
        "risk_label": risk_label,
        "confidence": confidence,
        "recommendation": recommendation,
        "timestamp": timestamp,
        "model_used": metadata["best_model"],
        "input": values,
        "patient_name": patient_name,
    })


@app.route("/api/history", methods=["GET"])
def history():
    limit = request.args.get("limit", 50, type=int)
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return jsonify({"success": True, "history": [dict(r) for r in rows]})


@app.route("/api/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    db = get_db()
    if q:
        rows = db.execute(
            """SELECT * FROM predictions
               WHERE patient_name LIKE ? OR risk_label LIKE ? OR created_at LIKE ?
               ORDER BY id DESC""",
            (f"%{q}%", f"%{q}%", f"%{q}%"),
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return jsonify({"success": True, "results": [dict(r) for r in rows]})


@app.route("/api/stats", methods=["GET"])
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    high = db.execute("SELECT COUNT(*) c FROM predictions WHERE risk_label='High Risk'").fetchone()["c"]
    low = db.execute("SELECT COUNT(*) c FROM predictions WHERE risk_label='Low Risk'").fetchone()["c"]
    avg_conf_row = db.execute("SELECT AVG(confidence) a FROM predictions").fetchone()
    avg_conf = round(avg_conf_row["a"], 2) if avg_conf_row["a"] else 0

    return jsonify({
        "success": True,
        "total_predictions": total,
        "high_risk_count": high,
        "low_risk_count": low,
        "average_confidence": avg_conf,
        "model_accuracies": metadata["all_results"],
        "best_model": metadata["best_model"],
    })


@app.route("/api/delete/<int:pred_id>", methods=["DELETE"])
def delete_prediction(pred_id):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    db.commit()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
