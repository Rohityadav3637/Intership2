"""
check_db.py
-----------
Quick utility to inspect the last 5 rows in the predictions table.

Usage (from repo root):
    python src/inference_server/check_db.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "inference_log.db"

if not DB_PATH.exists():
    print(f"[ERROR] Database not found at: {DB_PATH}")
    print("Make sure the FastAPI server has received at least one /predict request.")
    raise SystemExit(1)

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row   # allow column-name access

rows = con.execute(
    "SELECT id, timestamp, image_filename, detections, is_uncertain "
    "FROM predictions "
    "ORDER BY id DESC "
    "LIMIT 5"
).fetchall()

con.close()

if not rows:
    print("No predictions logged yet.")
    raise SystemExit(0)

SEP = "-" * 70
print(f"\n{'LAST 5 PREDICTIONS':^70}")
print(SEP)

for row in rows:
    detections = json.loads(row["detections"])
    uncertain_flag = "[!] UNCERTAIN" if row["is_uncertain"] else "[OK] CONFIDENT"

    print(f"  ID            : {row['id']}")
    print(f"  Timestamp     : {row['timestamp']}")
    print(f"  Image file    : {row['image_filename']}")
    print(f"  Status        : {uncertain_flag}")
    print(f"  Num detections: {len(detections)}")

    if detections:
        print("  Detections    :")
        for d in detections:
            bb = d["bbox"]
            print(
                f"    - {d['class_name']:<12}  conf={d['confidence']:.4f}  "
                f"bbox=({bb['x_min']:.0f},{bb['y_min']:.0f})"
                f"→({bb['x_max']:.0f},{bb['y_max']:.0f})"
            )
    print(SEP)
