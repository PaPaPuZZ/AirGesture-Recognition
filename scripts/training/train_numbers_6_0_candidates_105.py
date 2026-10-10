"""
NUMBERS 6-0 — 105 RECORDING RETRAINING

Training dataset:
    numbers_6_0_105_combined_training_raw_data.csv

Expected class counts:
    6: 15
    7: 15
    8: 20
    9: 25
    0: 30

Total:
    105 recordings

Raw recording:
    100 samples x 6 MPU6050 channels

Feature extraction:
    4 temporal segments
    x 6 channels
    x 9 statistics
    = 216 features

Exact features per channel per segment:
    1. mean
    2. std
    3. min
    4. max
    5. range
    6. RMS
    7. average absolute change
    8. maximum absolute change
    9. total movement

Candidate models:
    Random Forest
    Logistic Regression
    RBF SVM

Model selection priority:
    1. OOF balanced accuracy
    2. OOF macro F1
    3. OOF accuracy

No class_weight is used.

The existing 50-recording model is NOT overwritten.
This script saves a NEW 105-recording model.
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


# ============================================================
# FILES
# ============================================================

DATA_FILE = (
    "numbers_6_0_105_combined_training_raw_data.csv"
)

MODEL_FILE = (
    "mpu6050_numbers_6_0_best_model_105.pkl"
)

COMPARISON_FILE = (
    "numbers_6_0_105_model_comparison.csv"
)

OOF_FILE = (
    "numbers_6_0_105_oof_predictions.csv"
)

CONFUSION_FILE = (
    "numbers_6_0_105_oof_confusion_matrix.csv"
)

SUMMARY_FILE = (
    "numbers_6_0_105_training_summary.txt"
)


# ============================================================
# DATA CONFIGURATION
# ============================================================

NUM_SAMPLES = 100
NUM_CHANNELS = 6

NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

FEATURES_PER_CHANNEL_PER_SEGMENT = 9

EXPECTED_FEATURES = (
    NUM_SEGMENTS
    * NUM_CHANNELS
    * FEATURES_PER_CHANNEL_PER_SEGMENT
)

CLASSES = [
    "0",
    "6",
    "7",
    "8",
    "9",
]

EXPECTED_COUNTS = {
    "0": 30,
    "6": 15,
    "7": 15,
    "8": 20,
    "9": 25,
}

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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_segment_features(segment):

    segment = np.asarray(
        segment,
        dtype=float
    )

    if segment.shape != (
        SAMPLES_PER_SEGMENT,
        NUM_CHANNELS
    ):

        raise ValueError(
            f"Expected segment shape "
            f"({SAMPLES_PER_SEGMENT}, "
            f"{NUM_CHANNELS}), "
            f"got {segment.shape}"
        )

    features = []

    for channel in range(
        NUM_CHANNELS
    ):

        x = segment[
            :,
            channel
        ]

        dx = np.diff(
            x
        )

        features.extend(
            [
                np.mean(x),

                np.std(x),

                np.min(x),

                np.max(x),

                np.max(x)
                - np.min(x),

                np.sqrt(
                    np.mean(
                        x ** 2
                    )
                ),

                np.mean(
                    np.abs(
                        dx
                    )
                ),

                np.max(
                    np.abs(
                        dx
                    )
                ),

                np.sum(
                    np.abs(
                        dx
                    )
                ),
            ]
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != 54:

        raise RuntimeError(
            f"Expected 54 features "
            f"per segment, "
            f"got {features.size}."
        )

    return features


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
            f"({NUM_SAMPLES}, "
            f"{NUM_CHANNELS}), "
            f"got {recording.shape}"
        )

    features = []

    for segment_index in range(
        NUM_SEGMENTS
    ):

        start = (
            segment_index
            * SAMPLES_PER_SEGMENT
        )

        end = (
            start
            + SAMPLES_PER_SEGMENT
        )

        segment = recording[
            start:end,
            :
        ]

        features.extend(
            extract_segment_features(
                segment
            )
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_FEATURES} "
            f"features, "
            f"got {features.size}."
        )

    if not np.all(
        np.isfinite(
            features
        )
    ):

        raise RuntimeError(
            "Feature vector contains "
            "NaN or infinity."
        )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    path = Path(
        DATA_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        path
    )

    print()
    print("=" * 72)
    print("DATASET CHECK")
    print("=" * 72)

    print(
        "CSV shape:",
        df.shape
    )

    if "actual" not in df.columns:

        raise RuntimeError(
            "Missing 'actual' "
            "label column."
        )

    labels = (
        df["actual"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.strip()
    )

    y = np.asarray(
        labels.tolist(),
        dtype=str
    )

    print()
    print(
        "Class counts:"
    )

    for digit in CLASSES:

        count = int(
            np.sum(
                y == digit
            )
        )

        expected = (
            EXPECTED_COUNTS[
                digit
            ]
        )

        print(
            f"{digit}: "
            f"{count}/"
            f"{expected}"
        )

        if count != expected:

            raise RuntimeError(
                f"Expected "
                f"{expected} samples "
                f"for {digit}, "
                f"found {count}."
            )

    unexpected = sorted(
        set(y)
        - set(CLASSES)
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected labels: "
            + str(unexpected)
        )

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(
            "value_"
        )
    ]

    def sensor_sort_key(column):

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

    sensor_columns = sorted(
        sensor_columns,
        key=sensor_sort_key
    )

    print()
    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    if len(
        sensor_columns
    ) != 600:

        raise RuntimeError(
            f"Expected 600 sensor columns, "
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

    print(
        "Missing values:",
        missing
    )

    if missing != 0:

        raise RuntimeError(
            "Dataset contains "
            "missing values."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise RuntimeError(
            "Dataset contains "
            "non-finite values."
        )

    recordings = (
        values.reshape(
            len(df),
            NUM_SAMPLES,
            NUM_CHANNELS
        )
    )

    print(
        "Recording tensor:",
        recordings.shape
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
        "Exact duplicates:",
        duplicates
    )

    if duplicates != 0:

        raise RuntimeError(
            "Exact duplicate "
            "recordings detected."
        )

    print()
    print(
        "DATASET CHECK: PASS"
    )

    return (
        df,
        recordings,
        y
    )


# ============================================================
# BUILD 216-FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    recordings
):

    print()
    print("=" * 72)
    print("FEATURE EXTRACTION")
    print("=" * 72)

    rows = []

    for index, recording in enumerate(
        recordings,
        start=1
    ):

        features = (
            extract_temporal_features(
                recording
            )
        )

        rows.append(
            features
        )

        if (
            index % 10 == 0
            or index == len(recordings)
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(recordings)}"
            )

    X = np.asarray(
        rows,
        dtype=float
    )

    print()
    print(
        "Feature matrix shape:",
        X.shape
    )

    expected_shape = (
        len(recordings),
        EXPECTED_FEATURES
    )

    print(
        "Expected shape:",
        expected_shape
    )

    if X.shape != expected_shape:

        raise RuntimeError(
            "Feature matrix shape "
            "does not match."
        )

    if not np.all(
        np.isfinite(
            X
        )
    ):

        raise RuntimeError(
            "Feature matrix contains "
            "NaN or infinity."
        )

    print(
        "FEATURE EXTRACTION: PASS"
    )

    return X


# ============================================================
# MODELS
# ============================================================

def build_models():

    random_forest = (
        RandomForestClassifier(
            n_estimators=750,
            random_state=42,
            n_jobs=-1
        )
    )

    logistic_regression = (
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        C=1.0,
                        random_state=42
                    )
                ),
            ]
        )
    )

    rbf_svm = (
        Pipeline(
            [
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
                        random_state=42
                    )
                ),
            ]
        )
    )

    return {
        "Random Forest": (
            random_forest
        ),

        "Logistic Regression": (
            logistic_regression
        ),

        "RBF SVM": (
            rbf_svm
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_models(
    X,
    y
):

    print()
    print("=" * 72)
    print(
        "5-FOLD STRATIFIED "
        "CROSS-VALIDATION"
    )
    print("=" * 72)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": (
            "accuracy"
        ),

        "balanced_accuracy": (
            "balanced_accuracy"
        ),

        "macro_f1": (
            "f1_macro"
        ),
    }

    comparison_rows = []

    fitted_models = {}

    predictions = {}

    candidates = (
        build_models()
    )

    for (
        name,
        model
    ) in candidates.items():

        print()
        print("-" * 72)
        print(name)
        print("-" * 72)

        result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise"
        )

        mean_accuracy = float(
            np.mean(
                result[
                    "test_accuracy"
                ]
            )
        )

        mean_balanced = float(
            np.mean(
                result[
                    "test_balanced_accuracy"
                ]
            )
        )

        mean_macro_f1 = float(
            np.mean(
                result[
                    "test_macro_f1"
                ]
            )
        )

        oof = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=-1
        )

        oof = np.asarray(
            oof,
            dtype=str
        )

        oof_accuracy = float(
            accuracy_score(
                y,
                oof
            )
        )

        oof_balanced = float(
            balanced_accuracy_score(
                y,
                oof
            )
        )

        oof_macro_f1 = float(
            f1_score(
                y,
                oof,
                average="macro"
            )
        )

        print(
            "Mean CV accuracy:",
            f"{mean_accuracy:.4f}"
        )

        print(
            "Mean CV balanced accuracy:",
            f"{mean_balanced:.4f}"
        )

        print(
            "Mean CV macro F1:",
            f"{mean_macro_f1:.4f}"
        )

        print()
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

        comparison_rows.append(
            {
                "model": name,

                "mean_cv_accuracy": (
                    mean_accuracy
                ),

                "mean_cv_balanced_accuracy": (
                    mean_balanced
                ),

                "mean_cv_macro_f1": (
                    mean_macro_f1
                ),

                "oof_accuracy": (
                    oof_accuracy
                ),

                "oof_balanced_accuracy": (
                    oof_balanced
                ),

                "oof_macro_f1": (
                    oof_macro_f1
                ),
            }
        )

        predictions[
            name
        ] = oof

        print()
        print(
            "Fitting final "
            "full-dataset model..."
        )

        model.fit(
            X,
            y
        )

        fitted_models[
            name
        ] = model

        print(
            "Done."
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = (
        comparison.sort_values(
            by=[
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

    return (
        comparison,
        best_name,
        fitted_models[
            best_name
        ],
        predictions[
            best_name
        ],
    )


# ============================================================
# SAVE MODEL / RESULTS
# ============================================================

def save_results(

    df,
    y,
    comparison,
    best_name,
    best_model,
    oof_predictions
):

    accuracy = float(
        accuracy_score(
            y,
            oof_predictions
        )
    )

    balanced = float(
        balanced_accuracy_score(
            y,
            oof_predictions
        )
    )

    macro_f1 = float(
        f1_score(
            y,
            oof_predictions,
            average="macro"
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
        f"{accuracy:.4f}"
    )

    print(
        "OOF balanced accuracy:",
        f"{balanced:.4f}"
    )

    print(
        "OOF macro F1:",
        f"{macro_f1:.4f}"
    )

    print()
    print(
        "Classification report:"
    )

    report = classification_report(
        y,
        oof_predictions,
        labels=CLASSES,
        digits=4,
        zero_division=0
    )

    print(
        report
    )

    matrix = confusion_matrix(
        y,
        oof_predictions,
        labels=CLASSES
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{x}"
            for x in CLASSES
        ],
        columns=[
            f"pred_{x}"
            for x in CLASSES
        ]
    )

    oof_df = pd.DataFrame(
        {
            "actual": y,
            "prediction": (
                oof_predictions
            ),
            "correct": (
                y
                == oof_predictions
            ),
        }
    )

    if "trial_number" in df.columns:

        oof_df.insert(
            1,
            "trial_number",
            df[
                "trial_number"
            ].to_numpy()
        )

    if "dataset_source" in df.columns:

        oof_df.insert(
            2,
            "dataset_source",
            df[
                "dataset_source"
            ].to_numpy()
        )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    oof_df.to_csv(
        OOF_FILE,
        index=False
    )

    matrix_df.to_csv(
        CONFUSION_FILE
    )

    saved_model = {
        "model": best_model,

        "model_name": best_name,

        "classes": CLASSES,

        "num_samples": NUM_SAMPLES,

        "num_channels": NUM_CHANNELS,

        "num_segments": NUM_SEGMENTS,

        "samples_per_segment": (
            SAMPLES_PER_SEGMENT
        ),

        "features_per_channel_per_segment": (
            FEATURES_PER_CHANNEL_PER_SEGMENT
        ),

        "expected_features": (
            EXPECTED_FEATURES
        ),

        "feature_definition": (
            FEATURE_DEFINITION
        ),

        "training_file": (
            DATA_FILE
        ),

        "training_rows": int(
            len(df)
        ),

        "training_class_counts": (
            EXPECTED_COUNTS
        ),

        "cv_oof_accuracy": (
            accuracy
        ),

        "cv_balanced_accuracy": (
            balanced
        ),

        "cv_macro_f1": (
            macro_f1
        ),

        "model_selection_metric": (
            "OOF balanced accuracy, "
            "then macro F1, "
            "then accuracy"
        ),
    }

    with open(
        MODEL_FILE,
        "wb"
    ) as handle:

        pickle.dump(
            saved_model,
            handle
        )

    summary = []

    summary.append(
        "NUMBERS 6-0 — 105 RECORDING "
        "TRAINING SUMMARY\n"
    )

    summary.append(
        "=" * 65
        + "\n"
    )

    summary.append(
        f"Training file: "
        f"{DATA_FILE}\n"
    )

    summary.append(
        f"Rows: "
        f"{len(df)}\n"
    )

    summary.append(
        f"Class counts: "
        f"{EXPECTED_COUNTS}\n"
    )

    summary.append(
        "Features: "
        "4 segments x 6 channels "
        "x 9 = 216\n"
    )

    summary.append(
        f"Feature definition: "
        f"{FEATURE_DEFINITION}\n\n"
    )

    summary.append(
        "MODEL COMPARISON\n"
    )

    summary.append(
        comparison.to_string(
            index=False
        )
        + "\n\n"
    )

    summary.append(
        "SELECTED MODEL\n"
    )

    summary.append(
        f"{best_name}\n"
    )

    summary.append(
        f"OOF accuracy: "
        f"{accuracy:.4f}\n"
    )

    summary.append(
        f"OOF balanced accuracy: "
        f"{balanced:.4f}\n"
    )

    summary.append(
        f"OOF macro F1: "
        f"{macro_f1:.4f}\n\n"
    )

    summary.append(
        "CLASSIFICATION REPORT\n"
    )

    summary.append(
        report
        + "\n"
    )

    Path(
        SUMMARY_FILE
    ).write_text(
        "".join(
            summary
        )
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

    print()
    print(
        "Previous 50-recording "
        "model was NOT overwritten."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 6-0 — "
        "105 RECORDING RETRAINING"
    )
    print("=" * 72)

    print()
    print(
        "Training dataset:"
    )

    print(
        DATA_FILE
    )

    print()
    print(
        "Model candidates:"
    )

    print(
        "  Random Forest"
    )

    print(
        "  Logistic Regression"
    )

    print(
        "  RBF SVM"
    )

    print()
    print(
        "Selection priority:"
    )

    print(
        "  1. OOF balanced accuracy"
    )

    print(
        "  2. OOF macro F1"
    )

    print(
        "  3. OOF accuracy"
    )

    (
        df,
        recordings,
        y
    ) = load_dataset()

    X = build_feature_matrix(
        recordings
    )

    (
        comparison,
        best_name,
        best_model,
        best_predictions
    ) = evaluate_models(
        X,
        y
    )

    print()
    print("=" * 72)
    print(
        "FINAL MODEL COMPARISON"
    )
    print("=" * 72)

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    save_results(
        df,
        y,
        comparison,
        best_name,
        best_model,
        best_predictions
    )

    print()
    print(
        "105-RECORDING TRAINING COMPLETE."
    )


if __name__ == "__main__":

    main()
