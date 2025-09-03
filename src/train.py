import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np
from .config import DATA_PROCESSED, MODELS_DIR, FEATURE_COLUMNS, TARGET_COLUMN

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_PROCESSED / "features.csv"


def train_unsupervised(df: pd.DataFrame):
    """Train an IsolationForest model for anomaly detection"""
    X = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300, contamination=0.05, random_state=42
    )
    iso.fit(Xs)

    joblib.dump(scaler, MODELS_DIR / "scaler_unsup.joblib")
    joblib.dump(iso, MODELS_DIR / "isoforest.joblib")
    print("✅ Saved unsupervised IsolationForest + scaler.")

    # Produce anomaly scores for inspection
    normal_scores = iso.score_samples(Xs)
    anomaly_scores = (normal_scores.min() - normal_scores)  # invert
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min() + 1e-9
    )

    out = df.copy()
    out["anomaly_score"] = anomaly_scores
    out.to_csv(DATA_PROCESSED / "scored_unsupervised.csv", index=False)
    print(f"📄 Saved anomaly scores to {DATA_PROCESSED / 'scored_unsupervised.csv'}")

    # After computing anomaly_scores and if label exists
    if "label" in df.columns:
     import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    y_true = df["label"].values

    for thr in [0.5, 0.6, 0.7, 0.8]:
        y_pred = (out["anomaly_score"].values >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        print(f"thr={thr:.2f} P={p:.2f} R={r:.2f} F1={f:.2f}")



def train_supervised(df: pd.DataFrame):
    """Train a RandomForest classifier if labels exist"""
    if TARGET_COLUMN not in df.columns:
        print("⚠️ No labels found; skipping supervised model.")
        return

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Ensure labels are valid (binary or multi-class)
    if y.nunique() < 2:
        print("⚠️ Not enough label diversity; skipping supervised training.")
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(Xs, y)

    preds = clf.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, preds)
    print(f"🎯 Supervised AUC: {auc:.3f}")
    print(classification_report(y, preds > 0.5))

    joblib.dump(scaler, MODELS_DIR / "scaler_sup.joblib")
    joblib.dump(clf, MODELS_DIR / "rf_supervised.joblib")
    print("✅ Saved supervised RF + scaler.")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"❌ Could not find {DATA_FILE}. Run preprocess first.")

    df = pd.read_csv(DATA_FILE)

    print(f"📂 Loaded dataset with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")

    train_unsupervised(df)
    train_supervised(df)


if __name__ == "__main__":
    main()
