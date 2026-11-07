#!/usr/bin/env python3
"""
TRAIN A-Z RECOGNITION ACROSS THREE MPU6050 BOARD POSES

Input directory:
    data/letters/board_pose_study/

Supported poses:
    BOARD_FACE_UP
    BOARD_FACE_DOWN
    BOARD_FACE_VERTICAL

Initial expected dataset:
    3 poses x 26 letters x 5 trials = 390 recordings

Feature sets:
    raw_axis_216
    magnitude_72
    raw_plus_magnitude_288

Models:
    Logistic Regression
    RBF SVM
    Random Forest

Primary evaluation:
    leave-one-board-pose-out grouped evaluation

Random stratified CV is reported only as a reference.
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
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "letters" / "board_pose_study"
RESULTS_DIR = REPO_ROOT / "results" / "letters" / "board_pose_study"
MODELS_DIR = REPO_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "mpu6050_letters_a_z_board_pose_robust_best_model.pkl"
COMPARISON_FILE = RESULTS_DIR / "letters_a_z_board_pose_model_comparison.csv"
GROUPED_PREDICTIONS_FILE = RESULTS_DIR / "letters_a_z_board_pose_grouped_predictions.csv"
GROUPED_CONFUSION_FILE = RESULTS_DIR / "letters_a_z_board_pose_grouped_confusion_matrix.csv"
PER_POSE_FILE = RESULTS_DIR / "letters_a_z_board_pose_leave_one_out_metrics.csv"
SUMMARY_FILE = RESULTS_DIR / "letters_a_z_board_pose_training_summary.txt"

NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4

EXPECTED_RAW_FEATURES = 216
EXPECTED_MAG_FEATURES = 72
EXPECTED_COMBINED_FEATURES = 288

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXPECTED_POSES = [
    "BOARD_FACE_UP",
    "BOARD_FACE_DOWN",
    "BOARD_FACE_VERTICAL",
]

RANDOM_STATE = 42

FEATURE_MODES = [
    "raw_axis_216",
    "magnitude_72",
    "raw_plus_magnitude_288",
]


def normalize_label(value):
    return str(value).strip().upper()


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
                    max_iter=10000,
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
            n_estimators=1200,
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


def validate_file(df, path, reference_columns):
    required = [
        "actual",
        "orientation_id",
        "session_id",
        "participant_id",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing columns {missing}")

    labels = df["actual"].map(normalize_label)
    unexpected = sorted(set(labels) - set(LETTERS))
    if unexpected:
        raise RuntimeError(f"{path.name}: unexpected labels {unexpected}")

    sensor_columns = get_sensor_columns(df)
    if len(sensor_columns) != NUM_SAMPLES * NUM_CHANNELS:
        raise RuntimeError(
            f"{path.name}: expected 600 sensor columns, found {len(sensor_columns)}"
        )

    if reference_columns is not None and sensor_columns != reference_columns:
        raise RuntimeError(f"{path.name}: sensor-column order mismatch")

    numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    if int(numeric.isna().sum().sum()) != 0:
        raise RuntimeError(f"{path.name}: missing/non-numeric sensor values")

    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{path.name}: non-finite sensor values")

    return labels, sensor_columns, values


def load_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    files = sorted(DATA_DIR.glob("letters_A_Z_*.csv"))
    if not files:
        raise RuntimeError(f"No A-Z board-pose CSV files found in {DATA_DIR}")

    frames = []
    reference_columns = None

    for path in files:
        df = pd.read_csv(path)
        labels, sensor_columns, _ = validate_file(
            df,
            path,
            reference_columns,
        )

        if reference_columns is None:
            reference_columns = sensor_columns

        df = df.copy()
        df["actual"] = labels
        df["source_file"] = path.name
        frames.append(df)

        pose = str(df["orientation_id"].iloc[0])
        print(f"Loaded {path.name}: {len(df)} rows, pose={pose}")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    poses = sorted(combined["orientation_id"].astype(str).unique())
    missing_poses = sorted(set(EXPECTED_POSES) - set(poses))
    if missing_poses:
        raise RuntimeError(f"Missing required board poses: {missing_poses}")

    sensor_columns = get_sensor_columns(combined)
    numeric = combined[sensor_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)

    duplicates = int(pd.DataFrame(values).duplicated().sum())
    if duplicates:
        raise RuntimeError(f"Exact duplicate recordings found: {duplicates}")

    y = combined["actual"].map(normalize_label).to_numpy(dtype=str)
    groups = combined["orientation_id"].astype(str).to_numpy()
    sessions = combined["session_id"].astype(str).to_numpy()
    recordings = values.reshape(len(combined), NUM_SAMPLES, NUM_CHANNELS)

    print()
    print("Combined rows:", len(combined))
    print("Pose counts:")
    print(combined["orientation_id"].value_counts())
    print()
    print("Class counts:")
    print(pd.Series(y, name="actual").value_counts().sort_index())
    print("Exact duplicate recordings:", duplicates)

    return combined, recordings, y, groups, sessions


def grouped_predictions(model, X, y, groups):
    logo = LeaveOneGroupOut()
    return cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=logo,
        n_jobs=-1,
    )


def main():
    print("=" * 72)
    print("A-Z RECOGNITION ACROSS MPU6050 BOARD POSES")
    print("=" * 72)

    df, recordings, y, pose_groups, sessions = load_dataset()

    min_class_count = int(pd.Series(y).value_counts().min())
    random_splits = min(5, min_class_count)
    if random_splits < 2:
        raise RuntimeError("Not enough examples per class for cross-validation")

    random_cv = StratifiedKFold(
        n_splits=random_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    prediction_cache = {}
    fitted_models = {}
    feature_cache = {}

    for feature_mode in FEATURE_MODES:
        print()
        print("=" * 72)
        print("FEATURE MODE:", feature_mode)
        print("=" * 72)

        X = np.vstack([
            extract_feature_mode(recording, feature_mode)
            for recording in recordings
        ])
        feature_cache[feature_mode] = X
        print("Feature matrix:", X.shape)

        for model_name, model in build_models().items():
            print()
            print("-" * 72)
            print(model_name)
            print("-" * 72)

            random_pred = cross_val_predict(
                model,
                X,
                y,
                cv=random_cv,
                n_jobs=-1,
            )
            grouped_pred = grouped_predictions(
                model,
                X,
                y,
                pose_groups,
            )

            random_metrics = score_predictions(y, random_pred)
            grouped_metrics = score_predictions(y, grouped_pred)

            comparison_rows.append({
                "feature_mode": feature_mode,
                "model": model_name,
                "random_accuracy": random_metrics["accuracy"],
                "random_balanced_accuracy": random_metrics["balanced_accuracy"],
                "random_macro_f1": random_metrics["macro_f1"],
                "grouped_accuracy": grouped_metrics["accuracy"],
                "grouped_balanced_accuracy": grouped_metrics[
                    "balanced_accuracy"
                ],
                "grouped_macro_f1": grouped_metrics["macro_f1"],
            })

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

            prediction_cache[(feature_mode, model_name)] = grouped_pred
            model.fit(X, y)
            fitted_models[(feature_mode, model_name)] = model

    comparison = pd.DataFrame(comparison_rows).sort_values(
        [
            "grouped_balanced_accuracy",
            "grouped_macro_f1",
            "grouped_accuracy",
            "random_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_feature = str(comparison.iloc[0]["feature_mode"])
    selected_model_name = str(comparison.iloc[0]["model"])
    selected_model = fitted_models[(selected_feature, selected_model_name)]
    selected_predictions = np.asarray(
        prediction_cache[(selected_feature, selected_model_name)],
        dtype=str,
    )

    grouped_metrics = score_predictions(y, selected_predictions)

    print()
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False))

    print()
    print("=" * 72)
    print("SELECTED A-Z ORIENTATION-ROBUST PIPELINE")
    print("=" * 72)
    print("Feature mode:", selected_feature)
    print("Model:", selected_model_name)
    print(f"Grouped accuracy: {grouped_metrics['accuracy']:.4f}")
    print(
        "Grouped balanced accuracy:",
        f"{grouped_metrics['balanced_accuracy']:.4f}",
    )
    print(f"Grouped macro F1: {grouped_metrics['macro_f1']:.4f}")
    print()
    print(
        classification_report(
            y,
            selected_predictions,
            labels=LETTERS,
            digits=4,
            zero_division=0,
        )
    )

    per_pose_rows = []
    for pose in EXPECTED_POSES:
        mask = pose_groups == pose
        metrics = score_predictions(y[mask], selected_predictions[mask])
        per_pose_rows.append({
            "held_out_pose": pose,
            "rows": int(mask.sum()),
            **metrics,
        })

    per_pose = pd.DataFrame(per_pose_rows)
    print()
    print("PER-POSE RESULTS")
    print(per_pose.to_string(index=False))

    comparison.to_csv(COMPARISON_FILE, index=False)
    per_pose.to_csv(PER_POSE_FILE, index=False)

    predictions_df = pd.DataFrame({
        "actual": y,
        "prediction": selected_predictions,
        "correct": y == selected_predictions,
        "orientation_id": pose_groups,
        "session_id": sessions,
    })

    for column in [
        "participant_id",
        "dominant_hand",
        "writing_plane",
        "trial_number",
        "source_file",
    ]:
        if column in df.columns:
            predictions_df[column] = df[column].to_numpy()

    predictions_df.to_csv(GROUPED_PREDICTIONS_FILE, index=False)

    matrix = pd.DataFrame(
        confusion_matrix(y, selected_predictions, labels=LETTERS),
        index=[f"actual_{letter}" for letter in LETTERS],
        columns=[f"pred_{letter}" for letter in LETTERS],
    )
    matrix.to_csv(GROUPED_CONFUSION_FILE)

    selected_X = feature_cache[selected_feature]

    saved = {
        "model": selected_model,
        "model_name": selected_model_name,
        "feature_mode": selected_feature,
        "classes": LETTERS,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "num_segments": NUM_SEGMENTS,
        "expected_features": int(selected_X.shape[1]),
        "training_rows": int(len(df)),
        "supported_board_poses": EXPECTED_POSES,
        "grouped_accuracy": grouped_metrics["accuracy"],
        "grouped_balanced_accuracy": grouped_metrics["balanced_accuracy"],
        "grouped_macro_f1": grouped_metrics["macro_f1"],
    }

    with open(MODEL_FILE, "wb") as handle:
        pickle.dump(saved, handle)

    summary_lines = [
        "A-Z BOARD-POSE ROBUST TRAINING SUMMARY\n",
        "=" * 70 + "\n",
        f"Rows: {len(df)}\n",
        f"Feature mode: {selected_feature}\n",
        f"Model: {selected_model_name}\n",
        f"Grouped accuracy: {grouped_metrics['accuracy']:.4f}\n",
        (
            "Grouped balanced accuracy: "
            f"{grouped_metrics['balanced_accuracy']:.4f}\n"
        ),
        f"Grouped macro F1: {grouped_metrics['macro_f1']:.4f}\n\n",
        "FINAL COMPARISON\n",
        comparison.to_string(index=False),
        "\n\nPER-POSE RESULTS\n",
        per_pose.to_string(index=False),
        "\n\nCLASSIFICATION REPORT\n",
        classification_report(
            y,
            selected_predictions,
            labels=LETTERS,
            digits=4,
            zero_division=0,
        ),
    ]
    SUMMARY_FILE.write_text("".join(summary_lines))

    print()
    print("Saved model:", MODEL_FILE)
    print("Saved comparison:", COMPARISON_FILE)
    print("Saved predictions:", GROUPED_PREDICTIONS_FILE)
    print("Saved confusion matrix:", GROUPED_CONFUSION_FILE)
    print("Saved per-pose metrics:", PER_POSE_FILE)
    print("Saved summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
