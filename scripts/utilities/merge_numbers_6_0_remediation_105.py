"""
MERGE ORIGINAL 6-0 TRAINING DATA + TARGETED REMEDIATION DATA

Inputs:
    numbers_6_0_5second_training_raw_data.csv
    numbers_6_0_targeted_remediation_raw_data.csv

Expected counts before merge:
    Original:
        6: 10
        7: 10
        8: 10
        9: 10
        0: 10
        Total = 50

    Remediation:
        6: 5
        7: 5
        8: 10
        9: 15
        0: 20
        Total = 55

Expected merged counts:
    6: 15
    7: 15
    8: 20
    9: 25
    0: 30

Total:
    105 recordings

This script:
    - does NOT overwrite either source CSV
    - does NOT train a model
    - validates both source datasets
    - merges them into a NEW CSV
    - checks sensor integrity
    - checks exact duplicate recordings
    - checks acceleration magnitude
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data/numbers"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_FILE = DATA_DIR / "numbers_6_0_5second_training_raw_data.csv"
REMEDIATION_FILE = DATA_DIR / "numbers_6_0_targeted_remediation_raw_data.csv"
OUTPUT_FILE = DATA_DIR / "numbers_6_0_105_combined_training_raw_data.csv"


# ============================================================
# EXPECTED DATA
# ============================================================

NUM_SAMPLES = 100
NUM_CHANNELS = 6

EXPECTED_SENSOR_COLUMNS = (
    NUM_SAMPLES
    * NUM_CHANNELS
)

EXPECTED_ORIGINAL_COUNTS = {
    "6": 10,
    "7": 10,
    "8": 10,
    "9": 10,
    "0": 10,
}

EXPECTED_REMEDIATION_COUNTS = {
    "6": 5,
    "7": 5,
    "8": 10,
    "9": 15,
    "0": 20,
}

EXPECTED_MERGED_COUNTS = {
    "6": 15,
    "7": 15,
    "8": 20,
    "9": 25,
    "0": 30,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_labels(df):

    if "actual" not in df.columns:

        raise RuntimeError(
            "Dataset is missing "
            "'actual' column."
        )

    return (
        df["actual"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.strip()
    )


def get_sensor_columns(df):

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(
            "value_"
        )
    ]

    def sort_key(column):

        pieces = (
            column
            .replace(
                "value_",
                ""
            )
            .split("_")
        )

        return (
            int(pieces[0]),
            int(pieces[1]),
        )

    return sorted(
        sensor_columns,
        key=sort_key
    )


def validate_counts(
    labels,
    expected_counts,
    dataset_name
):

    print()
    print(
        f"{dataset_name} class counts:"
    )

    for digit, expected in (
        expected_counts.items()
    ):

        actual_count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"  {digit}: "
            f"{actual_count}/"
            f"{expected}"
        )

        if actual_count != expected:

            raise RuntimeError(
                f"{dataset_name}: "
                f"expected {expected} "
                f"recordings for {digit}, "
                f"found {actual_count}."
            )


def validate_dataset(
    df,
    expected_counts,
    dataset_name
):

    print()
    print("=" * 72)
    print(
        f"VALIDATING {dataset_name}"
    )
    print("=" * 72)

    print(
        "Shape:",
        df.shape
    )

    labels = normalize_labels(
        df
    )

    validate_counts(
        labels,
        expected_counts,
        dataset_name
    )

    sensor_columns = (
        get_sensor_columns(
            df
        )
    )

    print()
    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    if len(
        sensor_columns
    ) != EXPECTED_SENSOR_COLUMNS:

        raise RuntimeError(
            f"{dataset_name}: "
            f"expected "
            f"{EXPECTED_SENSOR_COLUMNS} "
            f"sensor columns, "
            f"found "
            f"{len(sensor_columns)}."
        )

    values = (
        df[
            sensor_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    missing = int(
        np.isnan(
            values
        ).sum()
    )

    finite = bool(
        np.all(
            np.isfinite(
                values
            )
        )
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "All finite:",
        finite
    )

    if missing != 0:

        raise RuntimeError(
            f"{dataset_name}: "
            "missing sensor values."
        )

    if not finite:

        raise RuntimeError(
            f"{dataset_name}: "
            "non-finite sensor values."
        )

    recordings = (
        values.reshape(
            len(df),
            NUM_SAMPLES,
            NUM_CHANNELS
        )
    )

    acceleration = recordings[
        :,
        :,
        0:3
    ]

    magnitude = np.sqrt(
        np.sum(
            acceleration ** 2,
            axis=2
        )
    )

    print()
    print(
        "Acceleration magnitude mean:",
        f"{magnitude.mean():.4f}",
        "m/s^2"
    )

    print(
        "Acceleration magnitude median:",
        f"{np.median(magnitude):.4f}",
        "m/s^2"
    )

    duplicates = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print(
        "Exact duplicate recordings:",
        duplicates
    )

    print()
    print(
        f"{dataset_name}: PASS"
    )

    return (
        labels,
        sensor_columns,
        values
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "MERGE NUMBERS 6-0 "
        "ORIGINAL + REMEDIATION"
    )
    print("=" * 72)

    original_path = Path(
        ORIGINAL_FILE
    )

    remediation_path = Path(
        REMEDIATION_FILE
    )

    if not original_path.exists():

        raise FileNotFoundError(
            f"Missing original file:\n"
            f"{ORIGINAL_FILE}"
        )

    if not remediation_path.exists():

        raise FileNotFoundError(
            f"Missing remediation file:\n"
            f"{REMEDIATION_FILE}"
        )

    original_df = pd.read_csv(
        original_path
    )

    remediation_df = pd.read_csv(
        remediation_path
    )

    (
        original_labels,
        original_sensor_columns,
        original_values
    ) = validate_dataset(
        original_df,
        EXPECTED_ORIGINAL_COUNTS,
        "ORIGINAL DATASET"
    )

    (
        remediation_labels,
        remediation_sensor_columns,
        remediation_values
    ) = validate_dataset(
        remediation_df,
        EXPECTED_REMEDIATION_COUNTS,
        "REMEDIATION DATASET"
    )

    print()
    print("=" * 72)
    print(
        "COLUMN COMPATIBILITY"
    )
    print("=" * 72)

    if list(
        original_sensor_columns
    ) != list(
        remediation_sensor_columns
    ):

        raise RuntimeError(
            "Sensor column ordering does "
            "not match between datasets."
        )

    print(
        "Sensor column ordering: MATCH"
    )

    # --------------------------------------------------------
    # Add source metadata
    # --------------------------------------------------------

    original_copy = (
        original_df.copy()
    )

    remediation_copy = (
        remediation_df.copy()
    )

    original_copy[
        "dataset_source"
    ] = "original_50"

    remediation_copy[
        "dataset_source"
    ] = "targeted_remediation_55"

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            original_copy,
            remediation_copy
        ],
        ignore_index=True
    )

    print()
    print("=" * 72)
    print(
        "MERGED DATASET"
    )
    print("=" * 72)

    print(
        "Rows:",
        len(combined_df)
    )

    expected_total = sum(
        EXPECTED_MERGED_COUNTS.values()
    )

    print(
        "Expected rows:",
        expected_total
    )

    if len(
        combined_df
    ) != expected_total:

        raise RuntimeError(
            f"Expected {expected_total} "
            f"merged rows, "
            f"found {len(combined_df)}."
        )

    merged_labels = (
        normalize_labels(
            combined_df
        )
    )

    validate_counts(
        merged_labels,
        EXPECTED_MERGED_COUNTS,
        "MERGED DATASET"
    )

    sensor_columns = (
        get_sensor_columns(
            combined_df
        )
    )

    values = (
        combined_df[
            sensor_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    missing = int(
        np.isnan(
            values
        ).sum()
    )

    finite = bool(
        np.all(
            np.isfinite(
                values
            )
        )
    )

    print()
    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "All finite:",
        finite
    )

    if missing != 0:

        raise RuntimeError(
            "Merged dataset contains "
            "missing sensor values."
        )

    if not finite:

        raise RuntimeError(
            "Merged dataset contains "
            "non-finite sensor values."
        )

    recordings = values.reshape(
        len(combined_df),
        NUM_SAMPLES,
        NUM_CHANNELS
    )

    accel = recordings[
        :,
        :,
        0:3
    ]

    magnitude = np.sqrt(
        np.sum(
            accel ** 2,
            axis=2
        )
    )

    print()
    print(
        "Overall acceleration "
        "magnitude mean:",
        f"{magnitude.mean():.4f}",
        "m/s^2"
    )

    print(
        "Overall acceleration "
        "magnitude median:",
        f"{np.median(magnitude):.4f}",
        "m/s^2"
    )

    # --------------------------------------------------------
    # Duplicate checking
    # --------------------------------------------------------

    duplicate_mask = (
        pd.DataFrame(
            values
        )
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    duplicate_count_after_first = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print()
    print(
        "Exact duplicate rows involved:",
        duplicate_rows
    )

    print(
        "Exact duplicates after first:",
        duplicate_count_after_first
    )

    if duplicate_count_after_first != 0:

        print()
        print(
            "WARNING:"
        )

        print(
            "At least one exact duplicate "
            "recording exists across the "
            "combined dataset."
        )

    else:

        print(
            "Duplicate check: PASS"
        )

    # --------------------------------------------------------
    # Source counts
    # --------------------------------------------------------

    print()
    print(
        "Source counts:"
    )

    print(
        combined_df[
            "dataset_source"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 72)
    print(
        "FINAL VALIDATION"
    )
    print("=" * 72)

    print(
        "Rows:",
        len(combined_df)
    )

    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "All finite:",
        finite
    )

    print(
        "Exact duplicates:",
        duplicate_count_after_first
    )

    print()
    print(
        "Final class counts:"
    )

    for digit in [
        "6",
        "7",
        "8",
        "9",
        "0"
    ]:

        count = int(
            (
                merged_labels == digit
            ).sum()
        )

        print(
            f"{digit}: {count}"
        )

    if (
        len(combined_df) == 105
        and len(sensor_columns) == 600
        and missing == 0
        and finite
        and duplicate_count_after_first == 0
    ):

        print()
        print(
            "MERGED DATASET VALIDATION: PASS"
        )

    else:

        print()
        print(
            "MERGED DATASET VALIDATION: "
            "CHECK REQUIRED"
        )

    print()
    print(
        "Saved merged dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Original and remediation "
        "files were NOT modified."
    )


if __name__ == "__main__":

    main()
