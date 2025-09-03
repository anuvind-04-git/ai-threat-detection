import csv
import json
import time
import requests
from pathlib import Path

# Simple file tailer for a CSV with the feature columns as headers
LOG_FILE = Path("data/raw/simlog.csv")  # change as needed
API = "http://127.0.0.1:1234/score"
FEATURES = ["bytes_out", "bytes_in", "duration_s", "hour", "is_private_ip"]


def follow(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.5)
                f.seek(where)
            else:
                # DictReader won’t parse mid-file; skip streaming append in this minimal demo
                pass


def main():
    for row in follow(LOG_FILE):
        payload = {k: float(row.get(k, 0) or 0) for k in FEATURES}
        try:
            r = requests.post(API, json=payload, timeout=2)
            j = r.json()
            print(payload, "→", j)
        except Exception as e:
            print("API error:", e)
        time.sleep(1)


if __name__ == "__main__":
    main()
