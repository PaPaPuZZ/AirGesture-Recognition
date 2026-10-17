#!/usr/bin/env python3
"""
TRAIN LETTERS M-Z CLASSIFIER

Input:
  data/letters/multisession/letters_M_Z_PaPaPuZZ_PaPaPuZZ_MZ_Session_01.csv

Expected:
  14 classes: M-Z
  10 recordings/class
  140 recordings total

Feature extractor:
  4 temporal segments x 6 channels x 9 statistics = 216 features

Candidates:
  Logistic Regression
  RBF SVM
  Random Forest

Selection priority:
  1. OOF balanced accuracy
  2. OOF macro F1
  3. OOF accuracy

The script preserves participant/session metadata in OOF output and
saves a new M-Z model without modifying prior A-L/D-L models.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    REPO_ROOT
    / "data/letters/multisession/"
      "letters_M_Z_PaPaPuZZ_PaPaPuZZ_MZ_Session_01.csv"
)

MODEL_FILE = (
    REPO_ROOT
    / "models/mpu6050_letters_M_Z_best_model.pkl"
)

RESULTS_DIR = REPO_ROOT / "results/letters"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = (
    RESULTS_DIR
    / "letters_M_Z_model_comparison.csv"
)

OOF_FILE = (
    RESULTS_DIR
    / "letters_M_Z_oof_predictions.csv"
)

CONFUSION_FILE = (
    RESULTS_DIR
    / "letters_M_Z_oof_confusion_matrix.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "letters_M_Z_training_summary.txt"
)


NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

EXPECTED_FEATURES = 216

CLASSES = list("MNOPQRSTUVWXYZ")

EXPECTED_COUNTS = {
    letter: 10
    for letter in CLASSES
}

RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_DEFINITION = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "avg_abs_change",
    "max_abs_change",
    "total_movement",
]


def normalize_label(value):
    return str(value).strip().upper()


def get_sensor_columns(df):
    columns = [
        column
        for column in df.columns
        if column.startswith("value_")
    ]

    def sort_key(column):
        sample, channel = (
            column
            .replace("value_", "")
            .split("_")
        )

        return int(sample), int(channel)

    return sorted(
        columns,
        key=sort_key
    )


def extract_temporal_features(recording):
    recording = np.asarray(
        recording,
        dtype=float
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS
    ):
        raise ValueError(
            f"Expected recording shape "
            f"({NUM_SAMPLES}, {NUM_CHANNELS}), "
            f"got {recording.shape}."
        )

    features = []

    for segment in np.array_split(
        recording,
        NUM_SEGMENTS,
        axis=0
    ):
        for channel in range(
            NUM_CHANNELS
        ):
            x = segment[:, channel]
            dx = np.diff(x)

            features.extend([
                np.mean(x),
                np.std(x),
                np.min(x),
                np.max(x),
                np.ptp(x),
                np.sqrt(
                    np.mean(x ** 2)
                ),
                np.mean(
                    np.abs(dx)
                ),
                np.max(
                    np.abs(dx)
                ),
                np.sum(
                    np.abs(dx)
                ),
            ])

    features = np.asarray(
        features,
        dtype=float
    )

    if features.shape != (
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            f"Expected 216 features, "
            f"got {features.shape}."
        )

    if not np.all(
        np.isfinite(features)
    ):
        raise RuntimeError(
            "Feature vector contains "
            "NaN or infinity."
        )

    return features


def build_models():
    return {
        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=7500,
                    C=1.0,
                    random_state=RANDOM_STATE,
                )
            ),
        ]),

        "RBF SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=RANDOM_STATE,
                )
            ),
        ]),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }


def main():
    print()
    print("=" * 72)
    print("LETTERS M-Z MODEL TRAINING")
    print("=" * 72)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            DATA_FILE
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Dataset:",
        DATA_FILE
    )

    print(
        "Shape:",
        df.shape
    )

    if "actual" not in df.columns:
        raise RuntimeError(
            "Missing actual column."
        )

    y = (
        df["actual"]
        .map(normalize_label)
        .to_numpy(dtype=str)
    )

    unexpected = sorted(
        set(y)
        - set(CLASSES)
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected classes: "
            f"{unexpected}"
        )

    print()
    print("Class counts:")

    for letter in CLASSES:
        count = int(
            np.sum(y == letter)
        )

        print(
            f"{letter}: "
            f"{count}/"
            f"{EXPECTED_COUNTS[letter]}"
        )

        if count != EXPECTED_COUNTS[
            letter
        ]:
            raise RuntimeError(
                f"Bad count for "
                f"{letter}: {count}"
            )

    sensor_columns = (
        get_sensor_columns(df)
    )

    if len(sensor_columns) != 600:
        raise RuntimeError(
            f"Expected 600 sensor columns, "
            f"found {len(sensor_columns)}."
        )

    numeric = (
        df[sensor_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    missing = int(
        numeric
        .isna()
        .sum()
        .sum()
    )

    if missing:
        raise RuntimeError(
            f"Missing/non-numeric values: "
            f"{missing}"
        )

    values = numeric.to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "Non-finite sensor values."
        )

    duplicates = int(
        pd.DataFrame(values)
        .duplicated()
        .sum()
    )

    print(
        "Exact duplicate recordings:",
        duplicates
    )

    if duplicates:
        raise RuntimeError(
            "Exact duplicate recordings detected."
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

    print()
    print("=" * 72)
    print("FEATURE EXTRACTION")
    print("=" * 72)

    X = np.vstack([
        extract_temporal_features(
            recording
        )
        for recording
        in recordings
    ])

    print(
        "Feature matrix:",
        X.shape
    )

    expected_shape = (
        len(df),
        EXPECTED_FEATURES
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"Expected {expected_shape}, "
            f"got {X.shape}."
        )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy":
            "accuracy",

        "balanced_accuracy":
            "balanced_accuracy",

        "macro_f1":
            "f1_macro",
    }

    comparison_rows = []
    fitted_models = {}
    predictions_by_model = {}

    print()
    print("=" * 72)
    print(
        "5-FOLD STRATIFIED "
        "CROSS-VALIDATION"
    )
    print("=" * 72)

    for name, model in (
        build_models().items()
    ):
        print()
        print("-" * 72)
        print(name)
        print("-" * 72)

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )

        predictions = np.asarray(
            predictions,
            dtype=str
        )

        oof_accuracy = float(
            accuracy_score(
                y,
                predictions
            )
        )

        oof_balanced = float(
            balanced_accuracy_score(
                y,
                predictions
            )
        )

        oof_macro_f1 = float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        comparison_rows.append({
            "model":
                name,

            "mean_cv_accuracy":
                float(
                    np.mean(
                        scores[
                            "test_accuracy"
                        ]
                    )
                ),

            "mean_cv_balanced_accuracy":
                float(
                    np.mean(
                        scores[
                            "test_balanced_accuracy"
                        ]
                    )
                ),

            "mean_cv_macro_f1":
                float(
                    np.mean(
                        scores[
                            "test_macro_f1"
                        ]
                    )
                ),

            "oof_accuracy":
                oof_accuracy,

            "oof_balanced_accuracy":
                oof_balanced,

            "oof_macro_f1":
                oof_macro_f1,
        })

        print(
            "OOF accuracy:",
            f"{oof_accuracy:.4f}"
        )

        print(
            "OOF balanced accuracy:",
            f"{oof_balanced:.4f}"
        )

        print(
            "OOF macro F1:",
            f"{oof_macro_f1:.4f}"
        )

        model.fit(
            X,
            y
        )

        fitted_models[
            name
        ] = model

        predictions_by_model[
            name
        ] = predictions

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (
        comparison
        .sort_values(
            [
                "oof_balanced_accuracy",
                "oof_macro_f1",
                "oof_accuracy",
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    best_name = str(
        comparison.iloc[0][
            "model"
        ]
    )

    best_model = (
        fitted_models[
            best_name
        ]
    )

    best_predictions = (
        predictions_by_model[
            best_name
        ]
    )

    print()
    print("=" * 72)
    print("FINAL MODEL COMPARISON")
    print("=" * 72)
    print(
        comparison.to_string(
            index=False
        )
    )

    best_accuracy = float(
        accuracy_score(
            y,
            best_predictions
        )
    )

    best_balanced = float(
        balanced_accuracy_score(
            y,
            best_predictions
        )
    )

    best_macro_f1 = float(
        f1_score(
            y,
            best_predictions,
            average="macro",
            zero_division=0,
        )
    )

    print()
    print("=" * 72)
    print("SELECTED MODEL")
    print("=" * 72)

    print(
        "Best model:",
        best_name
    )

    print(
        "OOF accuracy:",
        f"{best_accuracy:.4f}"
    )

    print(
        "OOF balanced accuracy:",
        f"{best_balanced:.4f}"
    )

    print(
        "OOF macro F1:",
        f"{best_macro_f1:.4f}"
    )

    print()
    print(
        classification_report(
            y,
            best_predictions,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    oof_df = pd.DataFrame({
        "actual":
            y,

        "prediction":
            best_predictions,

        "correct":
            y == best_predictions,
    })

    for column in [
        "participant_id",
        "session_id",
        "dominant_hand",
        "trial_number",
    ]:
        if column in df.columns:
            oof_df[
                column
            ] = df[
                column
            ].to_numpy()

    oof_df.to_csv(
        OOF_FILE,
        index=False
    )

    matrix = pd.DataFrame(
        confusion_matrix(
            y,
            best_predictions,
            labels=CLASSES
        ),
        index=[
            f"actual_{x}"
            for x in CLASSES
        ],
        columns=[
            f"pred_{x}"
            for x in CLASSES
        ],
    )

    matrix.to_csv(
        CONFUSION_FILE
    )

    saved = {
        "model":
            best_model,

        "model_name":
            best_name,

        "classes":
            CLASSES,

        "num_samples":
            NUM_SAMPLES,

        "num_channels":
            NUM_CHANNELS,

        "num_segments":
            NUM_SEGMENTS,

        "samples_per_segment":
            SAMPLES_PER_SEGMENT,

        "expected_features":
            EXPECTED_FEATURES,

        "feature_definition":
            FEATURE_DEFINITION,

        "training_file":
            str(DATA_FILE),

        "training_rows":
            int(len(df)),

        "training_class_counts": {
            letter:
                int(
                    np.sum(
                        y == letter
                    )
                )
            for letter
            in CLASSES
        },

        "oof_accuracy":
            best_accuracy,

        "oof_balanced_accuracy":
            best_balanced,

        "oof_macro_f1":
            best_macro_f1,
    }

    with open(
        MODEL_FILE,
        "wb"
    ) as handle:
        pickle.dump(
            saved,
            handle
        )

    summary = [
        "LETTERS M-Z TRAINING SUMMARY\n",
        "=" * 70 + "\n",
        f"Training file: {DATA_FILE}\n",
        f"Rows: {len(df)}\n",
        f"Selected model: {best_name}\n",
        f"OOF accuracy: {best_accuracy:.4f}\n",
        f"OOF balanced accuracy: {best_balanced:.4f}\n",
        f"OOF macro F1: {best_macro_f1:.4f}\n\n",
        "MODEL COMPARISON\n",
        comparison.to_string(
            index=False
        ),
        "\n\nCLASSIFICATION REPORT\n",
        classification_report(
            y,
            best_predictions,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        ),
    ]

    SUMMARY_FILE.write_text(
        "".join(summary)
    )

    print()
    print("=" * 72)
    print("SAVED OUTPUTS")
    print("=" * 72)

    print(
        "Model:",
        MODEL_FILE
    )

    print(
        "Comparison:",
        COMPARISON_FILE
    )

    print(
        "OOF predictions:",
        OOF_FILE
    )

    print(
        "Confusion matrix:",
        CONFUSION_FILE
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
