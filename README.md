````markdown
 AI-Driven Threat Detection 🔎🛡️

This repository contains an end-to-end AI-driven Threat Detection System.  
It trains an Isolation Forest model on security log features to identify and flag anomalous network activity.  
The project includes a complete pipeline from data preprocessing to a REST API and a web-based dashboard for real-time scoring.  

⚠️ **For educational use only. Not for production deployment without security review.**



 ✨ Key Components

- `src/preprocess.py` → Cleans raw log data and extracts relevant features.  
- `src/train.py` → Trains the unsupervised Isolation Forest model and an optional supervised Random Forest model for evaluation.  
- `src/api.py` → Flask-based REST API serving the trained model + simple web dashboard.  
- `src/static/index.html` → Dashboard UI for manual testing and visualization.  
- `src/stream_agent.py` → Simulates real-time log analysis by continuously feeding new data to the API.  
- `src/infer.py` → Command-line helper to score individual log entries.  


 📂 Project Structure

```
ai-threat-detection/
├─ .venv/                 # Python virtual environment
├─ data/
│  ├─ interim/            # Cleaned, intermediate data files
│  ├─ processed/          # Final ML-ready datasets
│  └─ raw/                # Original raw data (e.g., CSV logs)
├─ models/
│  ├─ isoforest.joblib    # Trained Isolation Forest model
│  └─ scaler_unsup.joblib # Data scaler
├─ src/
│  ├─ __init__.py         # Makes 'src' a Python package
│  ├─ api.py              # REST API + dashboard server
│  ├─ config.py           # Project configuration (paths, columns)
│  ├─ features.py         # Feature engineering helpers
│  ├─ infer.py            # CLI interface for inference
│  ├─ preprocess.py       # Data preprocessing script
│  ├─ stream_agent.py     # Real-time log streaming agent
│  └─ train.py            # Model training script
├─ .gitignore             # Ignore venv, models, etc.
├─ requirements.txt       # Project dependencies
└─ README.md              # Project overview and instructions
````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anuvind-04-git/ai-threat-detection.git
cd ai-threat-detection
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# On Windows PowerShell
.venv\Scripts\activate
# On Linux / Git Bash
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Preprocess dataset

```bash
python -m src.preprocess
```

### 2. Train anomaly detection models

```bash
python -m src.train
```

### 3. Run REST API + Dashboard

```bash
python -m src.api
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 4. Score a single row via CLI

```bash
python -m src.infer --row '{"bytes_out":1200,"bytes_in":800,"duration_s":12,"hour":10,"is_private_ip":1}'
```

### 5. Stream log scoring (real-time)

```bash
python -m src.stream_agent
```

---

## 📊 Demo Screenshot

<p align="center">
  <img src="docs/demo-dashboard.png" width="850" alt="Threat Detection Dashboard Demo"/>
</p>



---

## 🧪 Example Use Cases

* Detect login attempts from **public IPs at unusual hours**.
* Flag **abnormal byte transfer patterns** in network logs.
* Monitor simulated or real datasets in **real-time**.

---

## 🧩 Common Issues

* **All rows marked anomaly** → retrain model & check `scaler.joblib`.
* **Flask import error** → make sure virtual environment is activated.
* **CSV parse errors** → ensure dataset is UTF-8 encoded, comma-delimited.

---



