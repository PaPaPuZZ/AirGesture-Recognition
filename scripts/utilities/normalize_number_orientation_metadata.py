#!/usr/bin/env python3
"""
NORMALIZE NUMBER ORIENTATION-STUDY METADATA

The first three orientation sessions were originally labeled AXIS_X_LONG,
AXIS_Y_LONG, and AXIS_Z_LONG. Based on the physical setups actually used,
those labels should be interpreted as board poses:

    AXIS_X_LONG -> BOARD_FACE_VERTICAL
    AXIS_Y_LONG -> BOARD_FACE_UP
    AXIS_Z_LONG -> BOARD_FACE_DOWN

This utility does NOT overwrite the original CSV files. It creates normalized
copies in:

    data/numbers/orientation_study/normalized/

Raw sensor values are preserved exactly. Original metadata are retained in
original_* columns for traceability.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "numbers" / "orientation_study"
OUTPUT_DIR = SOURCE_DIR / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 100
NUM_CHANNELS = 6
EXPECTED_SENSOR_COLUMNS = NUM_SAMPLES * NUM_CHANNELS
DIGITS = [str(i) for i in range(10)]
EXPECTED_PER_DIGIT = 5

POSE_MAP = {
    "AXIS_X_LONG": {
        "orientation_id": "BOARD_FACE_VERTICAL",
        "sensor_mount_id": "MOUNT_FACE_VERTICAL",
        "session_suffix": "BOARD_FACE_VERTICAL_01",
        "note": "MPU6050 board mounted vertically/on edge.",
    },
    "AXIS_Y_LONG": {
        "orientation_id": "BOARD_FACE_UP",
        "sensor_mount_id": "MOUNT_FACE_UP",
        "session_suffix": "BOARD_FACE_UP_01",
        "note": "MPU6050 board horizontal and right-side up; historical baseline pose.",
    },
    "AXIS_Z_LONG": {
        "orientation_id": "BOARD_FACE_DOWN",
        "sensor_mount_id": "MOUNT_FACE_DOWN",
        "session_suffix": "BOARD_FACE_DOWN_01",
        "note": "MPU6050 board horizontal and upside down.",
    },
}


def normalize_label(value):
    return str(value).replace(".0", "").strip()


def get_sensor_columns(df):
    columns = [c for c in df.columns if c.startswith("value_")]

    def key(column):
        sample, channel = column.replace("value_", "").split("_")
        return int(sample), int(channel)

    return sorted(columns, key=key)


def identify_pose(df, path):
    candidates = [path.stem.upper()]

    for column in ["orientation_id", "sensor_mount_id", "session_id"]:
        if column in df.columns:
            candidates.extend(df[column].astype(str).str.upper().unique().tolist())

    joined = " ".join(candidates)

    matches = [key for key in POSE_MAP if key in joined]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify pose for {path.name}. Matches: {matches}"
        )

    return matches[0]


def validate(df, name):
    if "actual" not in df.columns:
        raise RuntimeError(f"{name}: missing actual column")

    labels = df["actual"].map(normalize_label)
    unexpected = sorted(set(labels) - set(DIGITS))

    if unexpected:
        raise RuntimeError(f"{name}: unexpected labels {unexpected}")

    if len(df) != len(DIGITS) * EXPECTED_PER_DIGIT:
        raise RuntimeError(f"{name}: expected 50 rows, found {len(df)}")

    for digit in DIGITS:
        count = int((labels == digit).sum())
        if count != EXPECTED_PER_DIGIT:
            raise RuntimeError(
                f"{name}: digit {digit} expected {EXPECTED_PER_DIGIT}, found {count}"
            )

    sensor_columns = get_sensor_columns(df)

    if len(sensor_columns) != EXPECTED_SENSOR_COLUMNS:
        raise RuntimeError(
            f"{name}: expected 600 sensor columns, found {len(sensor_columns)}"
        )

    numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().sum().sum())

    if missing:
        raise RuntimeError(f"{name}: missing/non-numeric sensor values: {missing}")

    values = numeric.to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name}: non-finite sensor values")

    duplicates = int(pd.DataFrame(values).duplicated().sum())

    if duplicates:
        raise RuntimeError(f"{name}: exact duplicate recordings: {duplicates}")

    return labels, sensor_columns


def preserve_original_column(df, column):
    original_column = f"original_{column}"

    if original_column not in df.columns:
        if column in df.columns:
            df[original_column] = df[column]
        else:
            df[original_column] = ""


def main():
    print("=" * 72)
    print("NORMALIZE NUMBER ORIENTATION-STUDY METADATA")
    print("=" * 72)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(SOURCE_DIR)

    source_files = sorted(
        path
        for path in SOURCE_DIR.glob("*.csv")
        if "normalized" not in path.stem.lower()
    )

    if not source_files:
        raise RuntimeError(f"No source CSV files found in {SOURCE_DIR}")

    processed = 0

    for path in source_files:
        df = pd.read_csv(path)

        try:
            pose_key = identify_pose(df, path)
        except RuntimeError as exc:
            print(f"SKIP: {path.name}: {exc}")
            continue

        labels, _ = validate(df, path.name)
        mapping = POSE_MAP[pose_key]

        normalized = df.copy()
        normalized["actual"] = labels

        for column in [
            "session_id",
            "study_type",
            "orientation_id",
            "sensor_mount_id",
            "pen_axis_alignment",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "condition_notes",
        ]:
            preserve_original_column(normalized, column)

        participant = (
            str(normalized["participant_id"].iloc[0])
            if "participant_id" in normalized.columns
            else "unknown_participant"
        )

        normalized["session_id"] = f"{participant}_{mapping['session_suffix']}"
        normalized["study_type"] = "board_mount_pose"
        normalized["orientation_id"] = mapping["orientation_id"]
        normalized["sensor_mount_id"] = mapping["sensor_mount_id"]
        normalized["pen_axis_alignment"] = "UNKNOWN"
        normalized["roll_deg"] = np.nan
        normalized["pitch_deg"] = np.nan
        normalized["yaw_deg"] = np.nan

        old_notes = (
            normalized["original_condition_notes"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        normalized["condition_notes"] = [
            f"{mapping['note']} Original note: {note}".strip()
            for note in old_notes
        ]

        output_file = (
            OUTPUT_DIR
            / (
                f"numbers_0_9_orientation_"
                f"{participant}_"
                f"{mapping['orientation_id']}.csv"
            )
        )

        validate(normalized, output_file.name)
        normalized.to_csv(output_file, index=False)

        print()
        print(f"Source: {path.name}")
        print(f"Original label: {pose_key}")
        print(f"Normalized pose: {mapping['orientation_id']}")
        print(f"Saved: {output_file}")

        processed += 1

    print()
    print(f"Normalized files created: {processed}")

    expected_outputs = {
        "BOARD_FACE_VERTICAL",
        "BOARD_FACE_UP",
        "BOARD_FACE_DOWN",
    }

    actual_outputs = {
        pd.read_csv(path, nrows=1)["orientation_id"].iloc[0]
        for path in OUTPUT_DIR.glob("*.csv")
    }

    missing_outputs = expected_outputs - actual_outputs

    if missing_outputs:
        print("WARNING: missing normalized poses:", sorted(missing_outputs))
    else:
        print("NORMALIZATION VALIDATION: PASS")

    print("Original source files were NOT modified.")


if __name__ == "__main__":
    main()
