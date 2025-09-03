import pandas as pd
from pathlib import Path
from src import config
from src.features import add_engineered_features, ensure_cols

def load_unsw_file(path: Path) -> pd.DataFrame:
    """Load UNSW dataset CSV into DataFrame"""
    return pd.read_csv(path)

def map_unsw_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW columns to generic features.
    Expected UNSW columns: sbytes, dbytes, dur, stime, srcip, label
    """
    mapped = pd.DataFrame()

    # Map traffic features
    mapped["bytes_out"] = df.get("sbytes", 0)
    mapped["bytes_in"] = df.get("dbytes", 0)
    mapped["duration_s"] = df.get("dur", 0)

    # Add timestamp if exists
    if "stime" in df.columns:
        mapped["ts"] = df["stime"]

    # Add source IP if exists
    if "srcip" in df.columns:
        mapped["ip"] = df["srcip"]

    # Label (if supervised)
    mapped["label"] = df.get("label", 0)

    return mapped

def main():
    # Paths
    train_path = config.DATA_RAW / "UNSW_NB15_training-set.csv"
    test_path = config.DATA_RAW / "UNSW_NB15_testing-set.csv"
    out_path = config.DATA_PROCESSED / "features.csv"

    print(f"Loading training data: {train_path}")
    df_train = load_unsw_file(train_path)

    print(f"Loading testing data: {test_path}")
    df_test = load_unsw_file(test_path)

    # Combine train + test
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Map to feature schema
    df_features = map_unsw_to_features(df_all)

    # Add engineered features
    df_features = add_engineered_features(df_features)

    # Ensure all expected columns exist
    df_features = ensure_cols(df_features, config.FEATURE_COLUMNS + [config.TARGET_COLUMN])

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_path, index=False)
    print(f"✅ Features saved to {out_path}")

if __name__ == "__main__":
    main()
