"""
MERGE 105-RECORDING NUMBERS 6-0 DATASET
WITH 20 NEW DIGIT-6 REMEDIATION RECORDINGS

Inputs:
    numbers_6_0_105_combined_training_raw_data.csv
    digit6_targeted_remediation_20_raw_data.csv

Expected original 105 counts:
    6: 15
    7: 15
    8: 20
    9: 25
    0: 30

Additional digit-6 recordings:
    6: 20

Expected merged counts:
    6: 35
    7: 15
    8: 20
    9: 25
    0: 30

Total:
    125 recordings

This script:
    - does NOT overwrite either source dataset
    - validates both inputs
    - creates a NEW 125-recording CSV
    - checks missing/non-finite values
    - checks exact duplicates
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

BASE_FILE = DATA_DIR / "numbers_6_0_105_combined_training_raw_data.csv"
DIGIT6_FILE = DATA_DIR / "digit6_targeted_remediation_20_raw_data.csv"
OUTPUT_FILE = DATA_DIR / "numbers_6_0_125_combined_training_raw_data.csv"


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 100
NUM_CHANNELS = 6

EXPECTED_SENSOR_COLUMNS = (
    NUM_SAMPLES
    * NUM_CHANNELS
)

EXPECTED_BASE_COUNTS = {
    "6": 15,
    "7": 15,
    "8": 20,
    "9": 25,
    "0": 30,
}

EXPECTED_DIGIT6_COUNTS = {
    "6": 20,
}

EXPECTED_FINAL_COUNTS = {
    "6": 35,
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
            "'actual' label column."
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

    columns = [
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
        columns,
        key=sort_key
    )


def check_counts(
    labels,
    expected_counts,
    name
):

    print()
    print(
        f"{name} class counts:"
    )

    for digit, expected in (
        expected_counts.items()
    ):

        count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"  {digit}: "
            f"{count}/"
            f"{expected}"
        )

        if count != expected:

            raise RuntimeError(
                f"{name}: expected "
                f"{expected} rows for "
                f"digit {digit}, "
                f"found {count}."
            )


def validate_dataset(
    df,
    expected_counts,
    name
):

    print()
    print("=" * 72)
    print(
        f"VALIDATING {name}"
    )
    print("=" * 72)

    print(
        "Shape:",
        df.shape
    )

    labels = normalize_labels(
        df
    )

    check_counts(
        labels,
        expected_counts,
        name
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
            f"{name}: expected "
            f"{EXPECTED_SENSOR_COLUMNS} "
            f"sensor columns, found "
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
            f"{name}: missing values found."
        )

    if not finite:

        raise RuntimeError(
            f"{name}: non-finite "
            "values found."
        )

    recordings = values.reshape(
        len(df),
        NUM_SAMPLES,
        NUM_CHANNELS
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
        "Exact duplicates:",
        duplicates
    )

    print()
    print(
        f"{name}: PASS"
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
        "105 + DIGIT-6 REMEDIATION"
    )
    print("=" * 72)

    base_path = Path(
        BASE_FILE
    )

    digit6_path = Path(
        DIGIT6_FILE
    )

    if not base_path.exists():

        raise FileNotFoundError(
            f"Missing base file:\n"
            f"{BASE_FILE}"
        )

    if not digit6_path.exists():

        raise FileNotFoundError(
            f"Missing digit-6 file:\n"
            f"{DIGIT6_FILE}"
        )

    base_df = pd.read_csv(
        base_path
    )

    digit6_df = pd.read_csv(
        digit6_path
    )

    (
        base_labels,
        base_sensor_columns,
        base_values
    ) = validate_dataset(
        base_df,
        EXPECTED_BASE_COUNTS,
        "105-RECORDING BASE DATASET"
    )

    (
        digit6_labels,
        digit6_sensor_columns,
        digit6_values
    ) = validate_dataset(
        digit6_df,
        EXPECTED_DIGIT6_COUNTS,
        "DIGIT-6 REMEDIATION DATASET"
    )

    # --------------------------------------------------------
    # Check column compatibility
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "COLUMN COMPATIBILITY"
    )
    print("=" * 72)

    if list(
        base_sensor_columns
    ) != list(
        digit6_sensor_columns
    ):

        raise RuntimeError(
            "Sensor column ordering "
            "does not match."
        )

    print(
        "Sensor column ordering: MATCH"
    )

    # --------------------------------------------------------
    # Preserve source metadata
    # --------------------------------------------------------

    base_copy = (
        base_df.copy()
    )

    digit6_copy = (
        digit6_df.copy()
    )

    if "dataset_source" not in (
        base_copy.columns
    ):

        base_copy[
            "dataset_source"
        ] = "base_105"

    else:

        base_copy[
            "dataset_source"
        ] = (
            base_copy[
                "dataset_source"
            ]
            .astype(str)
        )

    digit6_copy[
        "dataset_source"
    ] = (
        "digit6_remediation_20"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            base_copy,
            digit6_copy
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

    if len(
        combined_df
    ) != 125:

        raise RuntimeError(
            f"Expected 125 rows, "
            f"found {len(combined_df)}."
        )

    merged_labels = (
        normalize_labels(
            combined_df
        )
    )

    check_counts(
        merged_labels,
        EXPECTED_FINAL_COUNTS,
        "125-RECORDING MERGED DATASET"
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

    recordings = values.reshape(
        len(combined_df),
        NUM_SAMPLES,
        NUM_CHANNELS
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

    # --------------------------------------------------------
    # Duplicate checking across both sources
    # --------------------------------------------------------

    duplicate_count = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print()
    print(
        "Exact duplicate recordings:",
        duplicate_count
    )

    if duplicate_count == 0:

        print(
            "Duplicate check: PASS"
        )

    else:

        print(
            "WARNING: duplicates detected."
        )

    # --------------------------------------------------------
    # Source counts
    # --------------------------------------------------------

    print()
    print(
        "Dataset source counts:"
    )

    print(
        combined_df[
            "dataset_source"
        ]
        .value_counts()
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
        duplicate_count
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
        "0",
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
        len(combined_df) == 125
        and len(sensor_columns) == 600
        and missing == 0
        and finite
        and duplicate_count == 0
    ):

        print()
        print(
            "125-DATASET VALIDATION: PASS"
        )

    else:

        print()
        print(
            "125-DATASET VALIDATION: "
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
        "Source datasets were NOT modified."
    )


if __name__ == "__main__":

    main()
