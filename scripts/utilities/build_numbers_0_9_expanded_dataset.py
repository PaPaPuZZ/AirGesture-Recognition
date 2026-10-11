#!/usr/bin/env python3
"""
BUILD EXPANDED 0-9 TRAINING DATASET

Combines:
  data/numbers/numbers_0_9_combined_training_raw_data.csv
with every CSV in:
  data/numbers/multisession/

The original 215-recording dataset is preserved.
All multisession files are preserved.
A new expanded training dataset is created.

Legacy rows are tagged as:
  participant_id = PaPaPuZZ
  session_id     = legacy_pre_multisession

This lets future training keep participant/session metadata.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "numbers"
MULTI_DIR = DATA_DIR / "multisession"

BASE_FILE = DATA_DIR / "numbers_0_9_combined_training_raw_data.csv"
OUTPUT_FILE = DATA_DIR / "numbers_0_9_expanded_training_raw_data.csv"

NUM_SAMPLES = 100
NUM_CHANNELS = 6
EXPECTED_SENSOR_COLUMNS = 600
CLASSES = [str(i) for i in range(10)]

def normalize_label(v):
    return str(v).replace(".0", "").strip()

def sensor_columns(df):
    cols=[c for c in df.columns if c.startswith("value_")]
    def key(c):
        a,b=c.replace("value_","").split("_")
        return int(a),int(b)
    return sorted(cols,key=key)

def validate(df,name):
    if "actual" not in df.columns:
        raise RuntimeError(f"{name}: missing actual column")
    labels=df["actual"].map(normalize_label)
    bad=sorted(set(labels)-set(CLASSES))
    if bad:
        raise RuntimeError(f"{name}: unexpected labels {bad}")
    cols=sensor_columns(df)
    if len(cols)!=EXPECTED_SENSOR_COLUMNS:
        raise RuntimeError(f"{name}: expected 600 sensor columns, found {len(cols)}")
    numeric=df[cols].apply(pd.to_numeric,errors="coerce")
    if int(numeric.isna().sum().sum())!=0:
        raise RuntimeError(f"{name}: missing/non-numeric sensor values")
    values=numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name}: non-finite values")
    return labels,cols,values

def main():
    print("="*72)
    print("BUILD EXPANDED NUMBERS 0-9 TRAINING DATASET")
    print("="*72)

    if not BASE_FILE.exists():
        raise FileNotFoundError(BASE_FILE)
    if not MULTI_DIR.exists():
        raise FileNotFoundError(
            f"{MULTI_DIR} does not exist. Collect/push a multisession CSV first."
        )

    base=pd.read_csv(BASE_FILE)
    base_labels,base_cols,_=validate(base,"BASE 215 DATASET")

    base=base.copy()
    base["actual"]=base_labels
    if "participant_id" not in base.columns:
        base["participant_id"]="PaPaPuZZ"
    else:
        base["participant_id"]=base["participant_id"].fillna("PaPaPuZZ")
    if "session_id" not in base.columns:
        base["session_id"]="legacy_pre_multisession"
    else:
        base["session_id"]=base["session_id"].fillna("legacy_pre_multisession")
    if "dominant_hand" not in base.columns:
        base["dominant_hand"]="unknown"
    if "timestamp" not in base.columns:
        base["timestamp"]=""
    if "capture_seconds" not in base.columns:
        base["capture_seconds"]=np.nan

    session_files=sorted(MULTI_DIR.glob("*.csv"))
    if not session_files:
        raise RuntimeError(f"No CSV files found in {MULTI_DIR}")

    frames=[base]
    print(f"Base rows: {len(base)}")
    print(f"Multisession files found: {len(session_files)}")

    for path in session_files:
        df=pd.read_csv(path)
        labels,cols,_=validate(df,path.name)
        if list(cols)!=list(base_cols):
            raise RuntimeError(f"{path.name}: sensor-column ordering mismatch")
        df=df.copy()
        df["actual"]=labels
        if "dataset_source" not in df.columns:
            df["dataset_source"]="multisession"
        frames.append(df)
        print(f"  {path.name}: {len(df)} rows")

    combined=pd.concat(frames,ignore_index=True,sort=False)
    labels,cols,values=validate(combined,"EXPANDED DATASET")
    combined["actual"]=labels

    duplicate_count=int(pd.DataFrame(values).duplicated().sum())

    print()
    print("Final rows:",len(combined))
    print("Sensor columns:",len(cols))
    print("Exact duplicate recordings:",duplicate_count)

    print()
    print("Class counts:")
    for digit in CLASSES:
        print(f"{digit}: {int((labels==digit).sum())}")

    print()
    print("Participant counts:")
    print(combined["participant_id"].astype(str).value_counts())

    print()
    print("Session counts:")
    print(combined["session_id"].astype(str).value_counts())

    recordings=values.reshape(len(combined),NUM_SAMPLES,NUM_CHANNELS)
    mag=np.sqrt(np.sum(recordings[:,:,:3]**2,axis=2))
    print()
    print(f"Acceleration magnitude mean: {mag.mean():.4f} m/s^2")
    print(f"Acceleration magnitude median: {np.median(mag):.4f} m/s^2")

    if duplicate_count:
        raise RuntimeError(
            "Exact duplicate recordings were found across merged sources. "
            "Do not train until duplicates are reviewed."
        )

    combined.to_csv(OUTPUT_FILE,index=False)

    print()
    print("EXPANDED DATASET VALIDATION: PASS")
    print("Saved:",OUTPUT_FILE)
    print("Source datasets were NOT modified.")

if __name__=="__main__":
    main()
