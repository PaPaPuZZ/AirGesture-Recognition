#!/usr/bin/env python3
"""
TRAIN AND EVALUATE NUMBER RECOGNITION ACROSS THREE MPU6050 BOARD POSES

Input directory:
    data/numbers/orientation_study/normalized/

Expected poses:
    BOARD_FACE_UP
    BOARD_FACE_DOWN
    BOARD_FACE_VERTICAL

Expected dataset:
    3 poses x 10 digits x 5 trials = 150 recordings

Feature sets compared:
    raw_axis_216
        4 segments x 6 raw IMU channels x 9 statistics

    magnitude_72
        4 segments x 2 magnitude channels x 9 statistics
        channels = acceleration magnitude and gyroscope magnitude

    raw_plus_magnitude_288
        raw_axis_216 + magnitude_72

Candidate classifiers:
    Logistic Regression
    RBF SVM
    Random Forest

Evaluation:
    1. Ordinary 5-fold stratified CV as a reference.
    2. Leave-one-board-pose-out grouped evaluation.

Model selection is based primarily on grouped balanced accuracy, because the
engineering goal is recognition on a board pose that was not used for training.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
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
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "numbers" / "orientation_study" / "normalized"
RESULTS_DIR = REPO_ROOT / "results" / "numbers" / "orientation_study"
MODELS_DIR = REPO_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "mpu6050_numbers_board_pose_robust_best_model.pkl"
COMPARISON_FILE = RESULTS_DIR / "board_pose_model_comparison.csv"
GROUPED_OOF_FILE = RESULTS_DIR / "board_pose_grouped_oof_predictions.csv"
GROUPED_CONFUSION_FILE = RESULTS_DIR / "board_pose_grouped_confusion_matrix.csv"
PER_POSE_FILE = RESULTS_DIR / "board_pose_leave_one_out_metrics.csv"
SUMMARY_FILE = RESULTS_DIR / "board_pose_training_summary.txt"

NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
EXPECTED_RAW_FEATURES = 216
EXPECTED_MAG_FEATURES = 72
EXPECTED_COMBINED_FEATURES = 288

DIGITS = [str(i) for i in range(10)]
EXPECTED_POSES = [
    "BOARD_FACE_UP",
    "BOARD_FACE_DOWN",
    "BOARD_FACE_VERTICAL",
]

RANDOM_STATE = 42

STAT_NAMES = [
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
    return str(value).replace(".0", "").strip()


def get_sensor_columns(df):
    columns = [column for column in df.columns if column.startswith("value_")]

    def key(column):
        sample, channel = column.replace("value_", "").split("_")
        return int(sample), int(channel)

    return sorted(columns, key=key)


def segment_statistics(signal):
    signal = np.asarray(signal, dtype=float)
    delta = np.diff(signal)

    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.ptp(signal),
        np.sqrt(np.mean(signal ** 2)),
        np.mean(np.abs(delta)),
        np.max(np.abs(delta)),
        np.sum(np.abs(delta)),
    ]


def extract_raw_axis_features(recording):
    features = []

    for segment in np.array_split(recording, NUM_SEGMENTS, axis=0):
        for channel in range(NUM_CHANNELS):
            features.extend(segment_statistics(segment[:, channel]))

    result = np.asarray(features, dtype=float)

    if result.shape != (EXPECTED_RAW_FEATURES,):
        raise RuntimeError(f"Expected 216 raw-axis features, got {result.shape}")

    return result


def extract_magnitude_features(recording):
    acceleration_magnitude = np.linalg.norm(recording[:, 0:3], axis=1)
    gyro_magnitude = np.linalg.norm(recording[:, 3:6], axis=1)

    magnitude_recording = np.column_stack([
        acceleration_magnitude,
        gyro_magnitude,
    ])

    features = []

    for segment in np.array_split(magnitude_recording, NUM_SEGMENTS, axis=0):
        for channel in range(2):
            features.extend(segment_statistics(segment[:, channel]))

    result = np.asarray(features, dtype=float)

    if result.shape != (EXPECTED_MAG_FEATURES,):
        raise RuntimeError(f"Expected 72 magnitude features, got {result.shape}")

    return result


def extract_feature_mode(recording, mode):
    raw = extract_raw_axis_features(recording)
    magnitude = extract_magnitude_features(recording)

    if mode == "raw_axis_216":
        return raw

    if mode == "magnitude_72":
        return magnitude

    if mode == "raw_plus_magnitude_288":
        return np.concatenate([raw, magnitude])

    raise ValueError(f"Unknown feature mode: {mode}")


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=7500,
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def score_predictions(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
    }


def load_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Normalized orientation directory not found: {DATA_DIR}\n"
            "Run scripts/utilities/normalize_number_orientation_metadata.py first."
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise RuntimeError(f"No normalized CSV files found in {DATA_DIR}")

    frames = []
    reference_sensor_columns = None

    for path in files:
        df = pd.read_csv(path)

        required = ["actual", "orientation_id", "session_id"]
        missing_required = [column for column in required if column not in df.columns]

        if missing_required:
            raise RuntimeError(f"{path.name}: missing metadata {missing_required}")

        labels = df["actual"].map(normalize_label)
        unexpected = sorted(set(labels) - set(DIGITS))

        if unexpected:
            raise RuntimeError(f"{path.name}: unexpected labels {unexpected}")

        orientation_values = df["orientation_id"].astype(str).unique()

        if len(orientation_values) != 1:
            raise RuntimeError(
                f"{path.name}: expected one orientation, found {orientation_values}"
            )

        orientation = orientation_values[0]

        if orientation not in EXPECTED_POSES:
            print(f"SKIP unsupported normalized pose: {path.name}: {orientation}")
            continue

        for digit in DIGITS:
            count = int((labels == digit).sum())
            if count != 5:
                raise RuntimeError(
                    f"{path.name}: digit {digit} expected 5 rows, found {count}"
                )

        sensor_columns = get_sensor_columns(df)

        if len(sensor_columns) != 600:
            raise RuntimeError(
                f"{path.name}: expected 600 sensor columns, found {len(sensor_columns)}"
            )

        if reference_sensor_columns is None:
            reference_sensor_columns = sensor_columns
        elif sensor_columns != reference_sensor_columns:
            raise RuntimeError(f"{path.name}: sensor-column ordering mismatch")

        numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")

        if int(numeric.isna().sum().sum()):
            raise RuntimeError(f"{path.name}: missing/non-numeric sensor values")

        if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
            raise RuntimeError(f"{path.name}: non-finite sensor values")

        frame = df.copy()
        frame["actual"] = labels
        frame["source_file"] = path.name
        frames.append(frame)

        print(f"Loaded {path.name}: {len(frame)} rows, pose={orientation}")

    if not frames:
        raise RuntimeError("No supported normalized pose datasets were loaded")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    pose_counts = combined["orientation_id"].value_counts().to_dict()

    missing_poses = set(EXPECTED_POSES) - set(pose_counts)

    if missing_poses:
        raise RuntimeError(f"Missing required poses: {sorted(missing_poses)}")

    if len(combined) != 150:
        raise RuntimeError(f"Expected 150 total rows, found {len(combined)}")

    values = (
        combined[reference_sensor_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    duplicates = int(pd.DataFrame(values).duplicated().sum())

    if duplicates:
        raise RuntimeError(f"Exact duplicate recordings found: {duplicates}")

    recordings = values.reshape(len(combined), NUM_SAMPLES, NUM_CHANNELS)
    labels = combined["actual"].to_numpy(dtype=str)
    groups = combined["orientation_id"].astype(str).to_numpy()

    print()
    print("Combined rows:", len(combined))
    print("Pose counts:")
    print(combined["orientation_id"].value_counts())
    print("Class counts:")
    print(combined["actual"].value_counts().sort_index())
    print("Exact duplicate recordings:", duplicates)

    return combined, recordings, labels, groups


def leave_one_pose_metrics(model, features, labels, groups):
    rows = []

    for held_out_pose in sorted(np.unique(groups)):
        train_mask = groups != held_out_pose
        test_mask = groups == held_out_pose

        estimator = clone(model)
        estimator.fit(features[train_mask], labels[train_mask])
        predictions = estimator.predict(features[test_mask])
        metrics = score_predictions(labels[test_mask], predictions)

        rows.append({
            "held_out_pose": held_out_pose,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            **metrics,
        })

    return rows


def main():
    print("=" * 72)
    print("NUMBER RECOGNITION ACROSS MPU6050 BOARD POSES")
    print("=" * 72)

    metadata, recordings, labels, groups = load_dataset()

    feature_modes = [
        "raw_axis_216",
        "magnitude_72",
        "raw_plus_magnitude_288",
    ]

    random_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    group_cv = LeaveOneGroupOut()

    comparison_rows = []
    grouped_predictions_by_key = {}
    feature_matrices = {}
    per_pose_rows = []

    for feature_mode in feature_modes:
        print()
        print("=" * 72)
        print("FEATURE MODE:", feature_mode)
        print("=" * 72)

        features = np.vstack([
            extract_feature_mode(recording, feature_mode)
            for recording in recordings
        ])

        feature_matrices[feature_mode] = features

        print("Feature matrix:", features.shape)

        for model_name, model in build_models().items():
            print()
            print("-" * 72)
            print(model_name)
            print("-" * 72)

            random_predictions = cross_val_predict(
                model,
                features,
                labels,
                cv=random_cv,
                n_jobs=-1,
            )

            grouped_predictions = cross_val_predict(
                model,
                features,
                labels,
                groups=groups,
                cv=group_cv,
                n_jobs=-1,
            )

            random_metrics = score_predictions(labels, random_predictions)
            grouped_metrics = score_predictions(labels, grouped_predictions)

            print(
                "Random CV: "
                f"accuracy={random_metrics['accuracy']:.4f} "
                f"balanced={random_metrics['balanced_accuracy']:.4f} "
                f"macro_f1={random_metrics['macro_f1']:.4f}"
            )

            print(
                "Leave-one-pose-out: "
                f"accuracy={grouped_metrics['accuracy']:.4f} "
                f"balanced={grouped_metrics['balanced_accuracy']:.4f} "
                f"macro_f1={grouped_metrics['macro_f1']:.4f}"
            )

            key = (feature_mode, model_name)
            grouped_predictions_by_key[key] = grouped_predictions

            comparison_rows.append({
                "feature_mode": feature_mode,
                "model": model_name,
                "random_accuracy": random_metrics["accuracy"],
                "random_balanced_accuracy": random_metrics["balanced_accuracy"],
                "random_macro_f1": random_metrics["macro_f1"],
                "grouped_accuracy": grouped_metrics["accuracy"],
                "grouped_balanced_accuracy": grouped_metrics["balanced_accuracy"],
                "grouped_macro_f1": grouped_metrics["macro_f1"],
            })

            for row in leave_one_pose_metrics(model, features, labels, groups):
                per_pose_rows.append({
                    "feature_mode": feature_mode,
                    "model": model_name,
                    **row,
                })

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        [
            "grouped_balanced_accuracy",
            "grouped_macro_f1",
            "grouped_accuracy",
            "random_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best_feature_mode = str(comparison.iloc[0]["feature_mode"])
    best_model_name = str(comparison.iloc[0]["model"])
    best_key = (best_feature_mode, best_model_name)
    best_predictions = np.asarray(grouped_predictions_by_key[best_key], dtype=str)
    best_features = feature_matrices[best_feature_mode]
    best_model = build_models()[best_model_name]

    print()
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False))

    print()
    print("=" * 72)
    print("SELECTED ORIENTATION-ROBUST PIPELINE")
    print("=" * 72)
    print("Feature mode:", best_feature_mode)
    print("Model:", best_model_name)

    best_grouped_metrics = score_predictions(labels, best_predictions)
    print("Grouped accuracy:", f"{best_grouped_metrics['accuracy']:.4f}")
    print(
        "Grouped balanced accuracy:",
        f"{best_grouped_metrics['balanced_accuracy']:.4f}",
    )
    print("Grouped macro F1:", f"{best_grouped_metrics['macro_f1']:.4f}")

    print()
    print(classification_report(
        labels,
        best_predictions,
        labels=DIGITS,
        digits=4,
        zero_division=0,
    ))

    best_model.fit(best_features, labels)

    comparison.to_csv(COMPARISON_FILE, index=False)
    pd.DataFrame(per_pose_rows).to_csv(PER_POSE_FILE, index=False)

    grouped_oof = metadata.copy()
    grouped_oof["prediction"] = best_predictions
    grouped_oof["correct"] = labels == best_predictions
    grouped_oof["selected_feature_mode"] = best_feature_mode
    grouped_oof["selected_model"] = best_model_name
    grouped_oof.to_csv(GROUPED_OOF_FILE, index=False)

    confusion = pd.DataFrame(
        confusion_matrix(labels, best_predictions, labels=DIGITS),
        index=[f"actual_{digit}" for digit in DIGITS],
        columns=[f"pred_{digit}" for digit in DIGITS],
    )
    confusion.to_csv(GROUPED_CONFUSION_FILE)

    selected_per_pose = pd.DataFrame(per_pose_rows)
    selected_per_pose = selected_per_pose[
        (selected_per_pose["feature_mode"] == best_feature_mode)
        & (selected_per_pose["model"] == best_model_name)
    ]

    saved = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_mode": best_feature_mode,
        "classes": DIGITS,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "num_segments": NUM_SEGMENTS,
        "raw_stat_definition": STAT_NAMES,
        "training_rows": int(len(metadata)),
        "training_poses": sorted(np.unique(groups).tolist()),
        "grouped_accuracy": best_grouped_metrics["accuracy"],
        "grouped_balanced_accuracy": best_grouped_metrics["balanced_accuracy"],
        "grouped_macro_f1": best_grouped_metrics["macro_f1"],
        "per_pose_metrics": selected_per_pose.to_dict(orient="records"),
    }

    with open(MODEL_FILE, "wb") as handle:
        pickle.dump(saved, handle)

    summary_lines = [
        "MPU6050 BOARD-POSE ROBUST NUMBER TRAINING\n",
        "=" * 72 + "\n",
        f"Rows: {len(metadata)}\n",
        f"Poses: {sorted(np.unique(groups).tolist())}\n",
        f"Selected feature mode: {best_feature_mode}\n",
        f"Selected model: {best_model_name}\n",
        f"Grouped accuracy: {best_grouped_metrics['accuracy']:.4f}\n",
        f"Grouped balanced accuracy: {best_grouped_metrics['balanced_accuracy']:.4f}\n",
        f"Grouped macro F1: {best_grouped_metrics['macro_f1']:.4f}\n\n",
        "FINAL COMPARISON\n",
        comparison.to_string(index=False),
        "\n\nSELECTED PIPELINE PER-POSE METRICS\n",
        selected_per_pose.to_string(index=False),
        "\n\nCLASSIFICATION REPORT\n",
        classification_report(
            labels,
            best_predictions,
            labels=DIGITS,
            digits=4,
            zero_division=0,
        ),
    ]

    SUMMARY_FILE.write_text("".join(summary_lines))

    print()
    print("Saved model:", MODEL_FILE)
    print("Saved comparison:", COMPARISON_FILE)
    print("Saved grouped OOF predictions:", GROUPED_OOF_FILE)
    print("Saved grouped confusion matrix:", GROUPED_CONFUSION_FILE)
    print("Saved per-pose metrics:", PER_POSE_FILE)
    print("Saved summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
