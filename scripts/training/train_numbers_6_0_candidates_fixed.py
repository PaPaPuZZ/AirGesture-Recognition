"""
FIXED NUMBERS 6-0 MODEL TRAINING

Input:
    numbers_6_0_5second_training_raw_data.csv

Expected:
    50 recordings total
    10 each for:
        0, 6, 7, 8, 9

Raw recording:
    100 samples x 6 channels

Feature architecture:
    4 temporal segments
    x 6 channels
    x 9 features
    = 216 total features

Features per channel per segment:
    mean
    std
    min
    max
    range
    rms
    avg_abs_change
    max_abs_change
    total_movement

Candidate models:
    Random Forest
    Logistic Regression
    RBF SVM

IMPORTANT FIX:
    class_weight="balanced" has been removed.

Why:
    Your dataset is already exactly balanced:
        10 examples per class.

This avoids the sklearn error:
    "classes [0,6,7,8,9] are not in class_weight"
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
    "numbers_6_0_5second_training_raw_data.csv"
)

MODEL_FILE = (
    "mpu6050_numbers_6_0_best_model.pkl"
)

COMPARISON_FILE = (
    "numbers_6_0_candidate_model_comparison.csv"
)

OOF_FILE = (
    "numbers_6_0_oof_predictions.csv"
)

CONFUSION_FILE = (
    "numbers_6_0_oof_confusion_matrix.csv"
)

SUMMARY_FILE = (
    "numbers_6_0_training_summary.txt"
)


# ============================================================
# DATA / FEATURE CONFIGURATION
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

EXPECTED_CLASSES = [
    "0",
    "6",
    "7",
    "8",
    "9",
]

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

def extract_segment_features(
    segment
):

    segment = np.asarray(
        segment,
        dtype=float
    )

    expected_shape = (
        SAMPLES_PER_SEGMENT,
        NUM_CHANNELS
    )

    if segment.shape != expected_shape:

        raise ValueError(
            f"Expected segment shape "
            f"{expected_shape}, "
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

        mean = np.mean(
            x
        )

        std = np.std(
            x
        )

        minimum = np.min(
            x
        )

        maximum = np.max(
            x
        )

        value_range = (
            maximum
            - minimum
        )

        rms = np.sqrt(
            np.mean(
                x ** 2
            )
        )

        avg_abs_change = np.mean(
            np.abs(
                dx
            )
        )

        max_abs_change = np.max(
            np.abs(
                dx
            )
        )

        total_movement = np.sum(
            np.abs(
                dx
            )
        )

        features.extend(
            [
                mean,
                std,
                minimum,
                maximum,
                value_range,
                rms,
                avg_abs_change,
                max_abs_change,
                total_movement,
            ]
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != 54:

        raise RuntimeError(
            f"Segment produced "
            f"{features.size} features; "
            f"expected 54."
        )

    return features


def extract_temporal_features(
    recording
):

    recording = np.asarray(
        recording,
        dtype=float
    )

    expected_shape = (
        NUM_SAMPLES,
        NUM_CHANNELS
    )

    if recording.shape != expected_shape:

        raise ValueError(
            f"Expected recording shape "
            f"{expected_shape}, "
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

        segment_features = (
            extract_segment_features(
                segment
            )
        )

        features.extend(
            segment_features
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Recording produced "
            f"{features.size} features; "
            f"expected {EXPECTED_FEATURES}."
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
# LOAD / VALIDATE DATASET
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
            "Dataset is missing "
            "'actual' label column."
        )

    # Convert all labels explicitly
    # to normal Python-style strings.
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

    for class_name in EXPECTED_CLASSES:

        count = int(
            np.sum(
                y == class_name
            )
        )

        print(
            f"{class_name}: {count}"
        )

        if count != 10:

            raise RuntimeError(
                f"Expected 10 recordings "
                f"for class {class_name}, "
                f"found {count}."
            )

    unexpected = sorted(
        set(y)
        - set(EXPECTED_CLASSES)
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected classes found: "
            + str(unexpected)
        )

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(
            "value_"
        )
    ]

    def sensor_sort_key(
        column
    ):

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

    if len(sensor_columns) != 600:

        raise RuntimeError(
            f"Expected 600 sensor columns; "
            f"found {len(sensor_columns)}."
        )

    raw_values = (
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
            raw_values
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
            raw_values
        )
    ):

        raise RuntimeError(
            "Dataset contains "
            "non-finite values."
        )

    recordings = raw_values.reshape(
        len(df),
        NUM_SAMPLES,
        NUM_CHANNELS
    )

    print(
        "Recording tensor:",
        recordings.shape
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
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    recordings
):

    print()
    print("=" * 72)
    print("FEATURE EXTRACTION")
    print("=" * 72)

    feature_rows = []

    total = len(
        recordings
    )

    for i, recording in enumerate(
        recordings,
        start=1
    ):

        features = (
            extract_temporal_features(
                recording
            )
        )

        feature_rows.append(
            features
        )

        if (
            i % 10 == 0
            or i == total
        ):

            print(
                f"Processed "
                f"{i}/{total}"
            )

    X = np.asarray(
        feature_rows,
        dtype=float
    )

    print()
    print(
        "Feature matrix shape:",
        X.shape
    )

    expected_shape = (
        total,
        EXPECTED_FEATURES
    )

    print(
        "Expected shape:",
        expected_shape
    )

    if X.shape != expected_shape:

        raise RuntimeError(
            "Feature matrix shape "
            "is incorrect."
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
# CANDIDATE MODELS
# ============================================================

def build_candidates():

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = (
        RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RBF SVM
    # --------------------------------------------------------

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
        "Random Forest": random_forest,
        "Logistic Regression": logistic_regression,
        "RBF SVM": rbf_svm,
    }


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_models(
    X,
    y
):

    print()
    print("=" * 72)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 72)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    candidates = (
        build_candidates()
    )

    comparison_rows = []

    fitted_models = {}

    predictions_by_model = {}

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": (
            "balanced_accuracy"
        ),
        "macro_f1": "f1_macro",
    }

    for (
        model_name,
        model
    ) in candidates.items():

        print()
        print("-" * 72)
        print(
            model_name
        )
        print("-" * 72)

        # Setting error_score="raise"
        # gives a direct traceback if
        # anything is actually wrong.
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

        oof_predictions = (
            cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                n_jobs=-1
            )
        )

        oof_predictions = np.asarray(
            oof_predictions,
            dtype=str
        )

        oof_accuracy = float(
            accuracy_score(
                y,
                oof_predictions
            )
        )

        oof_balanced = float(
            balanced_accuracy_score(
                y,
                oof_predictions
            )
        )

        oof_macro_f1 = float(
            f1_score(
                y,
                oof_predictions,
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
                "model": model_name,
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

        predictions_by_model[
            model_name
        ] = (
            oof_predictions
        )

        print()
        print(
            "Fitting final model "
            "on all 50 recordings..."
        )

        model.fit(
            X,
            y
        )

        fitted_models[
            model_name
        ] = model

        print(
            "Done."
        )

    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison_df = (
        comparison_df.sort_values(
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

    best_model_name = str(
        comparison_df.iloc[0][
            "model"
        ]
    )

    best_model = (
        fitted_models[
            best_model_name
        ]
    )

    best_predictions = (
        predictions_by_model[
            best_model_name
        ]
    )

    return (
        comparison_df,
        best_model_name,
        best_model,
        best_predictions,
    )


# ============================================================
# SAVE EVERYTHING
# ============================================================

def save_results(
    df,
    y,
    comparison_df,
    best_model_name,
    best_model,
    predictions
):

    print()
    print("=" * 72)
    print("SELECTED MODEL")
    print("=" * 72)

    accuracy = float(
        accuracy_score(
            y,
            predictions
        )
    )

    balanced = float(
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    macro_f1 = float(
        f1_score(
            y,
            predictions,
            average="macro"
        )
    )

    print(
        "Best model:",
        best_model_name
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

    report = (
        classification_report(
            y,
            predictions,
            labels=EXPECTED_CLASSES,
            digits=4,
            zero_division=0
        )
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions,
        labels=EXPECTED_CLASSES
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{x}"
            for x in EXPECTED_CLASSES
        ],
        columns=[
            f"pred_{x}"
            for x in EXPECTED_CLASSES
        ]
    )

    # --------------------------------------------------------
    # OOF table
    # --------------------------------------------------------

    oof_df = pd.DataFrame(
        {
            "actual": y,
            "prediction": predictions,
            "correct": (
                y == predictions
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

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    oof_df.to_csv(
        OOF_FILE,
        index=False
    )

    confusion_df.to_csv(
        CONFUSION_FILE
    )

    # --------------------------------------------------------
    # Save model + metadata
    # --------------------------------------------------------

    saved_model = {
        "model": best_model,

        "model_name": (
            best_model_name
        ),

        "classes": (
            EXPECTED_CLASSES
        ),

        "num_samples": (
            NUM_SAMPLES
        ),

        "num_channels": (
            NUM_CHANNELS
        ),

        "num_segments": (
            NUM_SEGMENTS
        ),

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

        "training_rows": (
            int(
                len(df)
            )
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
    }

    with open(
        MODEL_FILE,
        "wb"
    ) as handle:

        pickle.dump(
            saved_model,
            handle
        )

    # --------------------------------------------------------
    # Summary text
    # --------------------------------------------------------

    lines = []

    lines.append(
        "NUMBERS 6-0 TRAINING SUMMARY\n"
    )

    lines.append(
        "=" * 60
        + "\n"
    )

    lines.append(
        f"Training file: "
        f"{DATA_FILE}\n"
    )

    lines.append(
        f"Rows: "
        f"{len(df)}\n"
    )

    lines.append(
        f"Classes: "
        f"{EXPECTED_CLASSES}\n"
    )

    lines.append(
        "Feature architecture: "
        "4 segments x 6 channels x "
        "9 features = 216\n"
    )

    lines.append(
        f"Features: "
        f"{FEATURE_DEFINITION}\n\n"
    )

    lines.append(
        "MODEL COMPARISON\n"
    )

    lines.append(
        comparison_df.to_string(
            index=False
        )
        + "\n\n"
    )

    lines.append(
        "SELECTED MODEL\n"
    )

    lines.append(
        f"{best_model_name}\n"
    )

    lines.append(
        f"OOF accuracy: "
        f"{accuracy:.4f}\n"
    )

    lines.append(
        f"OOF balanced accuracy: "
        f"{balanced:.4f}\n"
    )

    lines.append(
        f"OOF macro F1: "
        f"{macro_f1:.4f}\n\n"
    )

    lines.append(
        "CLASSIFICATION REPORT\n"
    )

    lines.append(
        report
    )

    Path(
        SUMMARY_FILE
    ).write_text(
        "".join(lines)
    )

    print()
    print("=" * 72)
    print("SAVED OUTPUTS")
    print("=" * 72)

    print(
        "Best model:",
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 6-0 "
        "FIXED CANDIDATE MODEL TRAINING"
    )
    print("=" * 72)

    print()
    print(
        "Dataset:"
    )

    print(
        DATA_FILE
    )

    print()
    print(
        "Models:"
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
        "class_weight:"
    )

    print(
        "  NONE"
    )

    print()
    print(
        "Reason:"
    )

    print(
        "  Dataset already has "
        "10 samples per class."
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
        comparison_df,
        best_model_name,
        best_model,
        predictions
    ) = evaluate_models(
        X,
        y
    )

    print()
    print("=" * 72)
    print("FINAL MODEL COMPARISON")
    print("=" * 72)

    print()
    print(
        comparison_df.to_string(
            index=False
        )
    )

    save_results(
        df,
        y,
        comparison_df,
        best_model_name,
        best_model,
        predictions
    )

    print()
    print(
        "TRAINING COMPLETE."
    )


if __name__ == "__main__":

    main()

