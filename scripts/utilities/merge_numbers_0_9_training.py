"""
MERGE FULL NUMBERS 0-9 TRAINING DATASET

Combines:

    numbers_1_5_50ms_training_raw_data.csv

with:

    numbers_6_0_125_combined_training_raw_data.csv

into:

    numbers_0_9_combined_training_raw_data.csv

This script does NOT train anything.

It checks:
    - labels
    - 600 sensor columns
    - numeric/finite data
    - compatible sensor-column ordering
    - duplicate recordings
    - acceleration magnitude
    - final 0-9 class counts

Source files are NOT modified.
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

FILE_1_5 = DATA_DIR / "numbers_1_5_50ms_training_raw_data.csv"
FILE_6_0 = DATA_DIR / "numbers_6_0_125_combined_training_raw_data.csv"
OUTPUT_FILE = DATA_DIR / "numbers_0_9_combined_training_raw_data.csv"


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 100
NUM_CHANNELS = 6

EXPECTED_SENSOR_COLUMNS = (
    NUM_SAMPLES
    * NUM_CHANNELS
)

EXPECTED_1_5 = {
    "1",
    "2",
    "3",
    "4",
    "5",
}

EXPECTED_6_0 = {
    "0",
    "6",
    "7",
    "8",
    "9",
}

ALL_CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_labels(df):

    if "actual" not in df.columns:

        raise RuntimeError(
            "Dataset does not contain "
            "an 'actual' label column."
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


def validate_dataset(
    df,
    allowed_classes,
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

    classes_found = set(
        labels.unique()
    )

    print()
    print(
        "Classes found:",
        sorted(
            classes_found
        )
    )

    unexpected = (
        classes_found
        - allowed_classes
    )

    if unexpected:

        raise RuntimeError(
            f"{name} contains "
            f"unexpected classes: "
            f"{sorted(unexpected)}"
        )

    print()
    print(
        "Class counts:"
    )

    for digit in sorted(
        allowed_classes
    ):

        count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"  {digit}: {count}"
        )

        if count == 0:

            raise RuntimeError(
                f"{name} contains "
                f"no recordings for {digit}."
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
            f"{name}: expected 600 "
            f"sensor columns, "
            f"found {len(sensor_columns)}."
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
            f"{name} contains "
            "missing sensor values."
        )


    if not finite:

        raise RuntimeError(
            f"{name} contains "
            "non-finite sensor values."
        )

    recordings = values.reshape(
        len(df),
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
        "Internal exact duplicates:",
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
        "BUILD FULL NUMBERS 0-9 "
        "TRAINING DATASET"
    )
    print("=" * 72)

    path_1_5 = Path(
        FILE_1_5
    )

    path_6_0 = Path(
        FILE_6_0
    )

    if not path_1_5.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{FILE_1_5}"
        )

    if not path_6_0.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{FILE_6_0}"
        )

    df_1_5 = pd.read_csv(
        path_1_5
    )

    df_6_0 = pd.read_csv(
        path_6_0
    )

    (
        labels_1_5,
        columns_1_5,
        values_1_5
    ) = validate_dataset(
        df_1_5,
        EXPECTED_1_5,
        "NUMBERS 1-5 DATASET"
    )

    (
        labels_6_0,
        columns_6_0,
        values_6_0
    ) = validate_dataset(
        df_6_0,
        EXPECTED_6_0,
        "NUMBERS 6-0 DATASET"
    )

    # --------------------------------------------------------
    # Sensor ordering
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "PIPELINE COMPATIBILITY"
    )
    print("=" * 72)

    if list(
        columns_1_5
    ) != list(
        columns_6_0
    ):

        raise RuntimeError(
            "Sensor-column ordering "
            "does not match between "
            "1-5 and 6-0 datasets."
        )

    print(
        "Sensor-column ordering: MATCH"
    )

    print(
        "Samples per recording:",
        NUM_SAMPLES
    )

    print(
        "Channels:",
        NUM_CHANNELS
    )

    # --------------------------------------------------------
    # Add dataset-source metadata
    # --------------------------------------------------------

    left = df_1_5.copy()

    right = df_6_0.copy()

    left[
        "dataset_source"
    ] = "numbers_1_5"

    right[
        "dataset_source"
    ] = "numbers_6_0_125"

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined = pd.concat(
        [
            left,
            right
        ],
        ignore_index=True,
        sort=False
    )

    labels = normalize_labels(
        combined
    )

    print()
    print("=" * 72)
    print(
        "FULL 0-9 DATASET"
    )
    print("=" * 72)

    print(
        "Total rows:",
        len(combined)
    )

    print()
    print(
        "Final class counts:"
    )

    for digit in ALL_CLASSES:

        count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"{digit}: {count}"
        )

        if count == 0:

            raise RuntimeError(
                f"No recordings found "
                f"for digit {digit}."
            )

    sensor_columns = (
        get_sensor_columns(
            combined
        )
    )

    values = (
        combined[
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

    duplicates = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
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

    print(
        "Exact duplicate recordings:",
        duplicates
    )

    recordings = values.reshape(
        len(combined),
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

    print()
    print(
        "Source counts:"
    )

    print(
        combined[
            "dataset_source"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 72)
    print(
        "FINAL 0-9 VALIDATION"
    )
    print("=" * 72)

    if (
        len(sensor_columns) == 600
        and missing == 0
        and finite
        and duplicates == 0
        and set(
            labels.unique()
        )
        == set(
            ALL_CLASSES
        )
    ):

        print(
            "FULL 0-9 DATASET: PASS"
        )

    else:

        print(
            "FULL 0-9 DATASET: "
            "CHECK REQUIRED"
        )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Source files were NOT modified."
    )


if __name__ == "__main__":

    main()
