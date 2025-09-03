from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from pathlib import Path
from .config import MODELS_DIR, FEATURE_COLUMNS, ANOMALY_THRESHOLD

# Use an absolute path to the static folder
BASE = Path(__file__).resolve().parent
STATIC_FOLDER = BASE / "static"

app = Flask(__name__, static_folder=STATIC_FOLDER)

# Load model + scaler
scaler = joblib.load(MODELS_DIR / "scaler_unsup.joblib")
model = joblib.load(MODELS_DIR / "isoforest.joblib")


@app.post("/score")
def score():
    data = request.get_json(force=True)
    x = np.array([[data.get(c, 0) for c in FEATURE_COLUMNS]], dtype=float)
    xs = scaler.transform(x)

    normal = model.score_samples(xs)[0]

    # Convert to anomaly score (invert + squash)
    anomaly_score = 1 / (1 + np.exp(float(normal)))
    is_anomaly = int(anomaly_score >= ANOMALY_THRESHOLD)

    return jsonify({
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly
    })


@app.get("/")
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
