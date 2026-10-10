#!/usr/bin/env python3
"""
compare_numbers_1_5_extractors_numeric.py

Read-only diagnostic for the numbers 1-5 pipeline.

What it does:
- Finds a likely numbers 1-5 CSV in the current folder.
- Loads one 100x6 IMU recording (600 sensor values).
- Computes two candidate feature definitions on the exact same recording:
    A) 4 segments x 6 channels x 9 stats = 216 features
    B) 8 global stats/channel + 7 segmented stats/channel = 216 features
- Compares the two vectors numerically.

It does NOT:
- connect to Arduino
- collect new data
- retrain a model
- overwrite any existing project files
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd

NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
EXPECTED_SENSOR_VALUES = NUM_SAMPLES * NUM_CHANNELS


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def training_extractor(recording: np.ndarray) -> np.ndarray:
    """Reference 216-feature extractor: 4 segments x 6 channels x 9 stats."""
    x = np.asarray(recording, dtype=float)
    if x.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise ValueError(f"Expected {(NUM_SAMPLES, NUM_CHANNELS)}, got {x.shape}")

    features = []
    for segment in np.array_split(x, NUM_SEGMENTS, axis=0):
        for ch in range(segment.shape[1]):
            v = segment[:, ch]
            dv = np.diff(v)
            features.extend([
                np.mean(v),
                np.std(v),
                np.min(v),
                np.max(v),
                np.ptp(v),
                np.sqrt(np.mean(v ** 2)),
                np.mean(np.abs(dv)),
                np.max(np.abs(dv)),
                np.sum(np.abs(dv)),
            ])

    return np.asarray(features, dtype=float)


def alternate_extractor(recording: np.ndarray) -> np.ndarray:
    """Alternate 216-feature definition seen in validation-style code."""
    x = np.asarray(recording, dtype=float)
    if x.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise ValueError(f"Expected {(NUM_SAMPLES, NUM_CHANNELS)}, got {x.shape}")

    features = []

    # 8 global statistics x 6 channels = 48
    for ch in range(NUM_CHANNELS):
        v = x[:, ch]
        idx = np.arange(len(v), dtype=float)
        slope = np.polyfit(idx, v, 1)[0] if len(v) > 1 else 0.0
        features.extend([
            np.mean(v),
            np.std(v),
            np.min(v),
            np.max(v),
            np.median(v),
            np.sqrt(np.mean(v ** 2)),
            np.ptp(v),
            slope,
        ])

    # 7 segment statistics x 6 channels x 4 segments = 168
    for segment in np.array_split(x, NUM_SEGMENTS, axis=0):
        for ch in range(NUM_CHANNELS):
            v = segment[:, ch]
            idx = np.arange(len(v), dtype=float)
            slope = np.polyfit(idx, v, 1)[0] if len(v) > 1 else 0.0
            features.extend([
                np.mean(v),
                np.std(v),
                np.min(v),
                np.max(v),
                np.median(v),
                np.sqrt(np.mean(v ** 2)),
                slope,
            ])

    return np.asarray(features, dtype=float)


def sorted_sensor_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if str(c).startswith("sensor_")]

    def key(col):
        try:
            return int(str(col).split("_")[-1])
        except Exception:
            return 10**9

    return sorted(cols, key=key)


def candidate_score(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if "numbers" in name or "number" in name:
        score += 5
    if "1_5" in name:
        score += 5
    if "training" in name:
        score += 5
    if "50ms" in name:
        score += 2
    if "holdout" in name:
        score -= 4
    if "result" in name or "prediction" in name or "summary" in name:
        score -= 6
    return score


def find_recording():
    candidates = []

    for filename in sorted(glob.glob("*.csv")):
        path = Path(filename)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        sensor_cols = sorted_sensor_columns(df)
        if len(sensor_cols) < EXPECTED_SENSOR_VALUES or len(df) == 0:
            continue

        sensor_cols = sensor_cols[:EXPECTED_SENSOR_VALUES]

        # Prefer a row labeled 1-5 when a recognizable label column exists.
        row_idx = 0
        label_col = None
        label_value = None

        for col in ("actual", "label", "class", "target"):
            if col in df.columns:
                label_col = col
                mask = df[col].astype(str).str.strip().isin(["1", "2", "3", "4", "5"])
                hits = np.flatnonzero(mask.to_numpy())
                if len(hits):
                    row_idx = int(hits[0])
                label_value = df.iloc[row_idx][col]
                break

        values = pd.to_numeric(
            df.iloc[row_idx][sensor_cols], errors="coerce"
        ).to_numpy(dtype=float)

        if values.size != EXPECTED_SENSOR_VALUES:
            continue
        if not np.all(np.isfinite(values)):
            continue

        recording = values.reshape(NUM_SAMPLES, NUM_CHANNELS)
        candidates.append(
            (candidate_score(path), path, row_idx, label_col, label_value, recording)
        )

    if not candidates:
        raise FileNotFoundError(
            "No CSV in this folder contained at least 600 sensor_* columns "
            "forming one 100x6 recording."
        )

    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return candidates[0], candidates


def main() -> None:
    section("NUMBERS 1-5 NUMERICAL FEATURE-EXTRACTOR CHECK")
    print("Read-only diagnostic: no Arduino, no retraining, no file overwrite.")

    selected, candidates = find_recording()
    score, path, row_idx, label_col, label_value, recording = selected

    section("CSV CANDIDATES")
    for i, item in enumerate(candidates[:12], 1):
        print(f"{i:2d}. {item[1].name} | score={item[0]}")

    section("SELECTED RECORDING")
    print(f"File:  {path.name}")
    print(f"Row:   {row_idx}")
    if label_col is not None:
        print(f"Label: {label_value} ({label_col})")
    print(f"Shape: {recording.shape}")

    accel_mag = np.sqrt(np.sum(recording[:, :3] ** 2, axis=1))
    print(f"Acceleration magnitude mean:   {np.mean(accel_mag):.6f}")
    print(f"Acceleration magnitude median: {np.median(accel_mag):.6f}")

    a = training_extractor(recording)
    b = alternate_extractor(recording)

    section("FEATURE COUNTS")
    print(f"Training/reference extractor: {len(a)}")
    print(f"Alternate extractor:          {len(b)}")
    print(f"Training features finite:     {np.all(np.isfinite(a))}")
    print(f"Alternate features finite:    {np.all(np.isfinite(b))}")

    section("NUMERICAL COMPARISON")
    if a.shape != b.shape:
        print("FAIL: feature-vector lengths differ.")
        print(f"Training shape:  {a.shape}")
        print(f"Alternate shape: {b.shape}")
    else:
        diff = np.abs(a - b)
        same = np.allclose(a, b, rtol=1e-9, atol=1e-10)
        print(f"Mean absolute difference: {np.mean(diff):.12g}")
        print(f"Max absolute difference:  {np.max(diff):.12g}")
        print(f"Numerically identical:     {same}")

        if not same:
            worst = np.argsort(diff)[::-1][:12]
            print("\nLargest differences:")
            for idx in worst:
                print(
                    f"  feature {idx:3d}: "
                    f"training={a[idx]: .9g}  "
                    f"alternate={b[idx]: .9g}  "
                    f"|diff|={diff[idx]:.9g}"
                )

    section("INTERPRETATION")
    print("The reference training extractor uses:")
    print("  4 temporal segments x 6 channels x 9 statistics = 216 features")
    print("  stats = mean, std, min, max, peak-to-peak, RMS,")
    print("          mean|diff|, max|diff|, sum|diff|")
    print()
    print("The alternate extractor uses a different mix/order of global and")
    print("segmented statistics, including median and slope.")
    print()
    print("Even if both vectors contain 216 numbers, they are NOT compatible")
    print("unless their values and ordering are identical.")
    print()
    print("Next step: make the final numbers 1-5 live/certification script call")
    print("the exact same extractor that was used to train its saved model.")


if __name__ == "__main__":
    main()

