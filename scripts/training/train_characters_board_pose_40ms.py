#!/usr/bin/env python3
"""
TRAIN 40 MS / 4 S CHARACTER RECOGNITION ACROSS MPU6050 BOARD POSES

Usage:
  python scripts/training/train_characters_board_pose_40ms.py --task numbers
  python scripts/training/train_characters_board_pose_40ms.py --task letters

Inputs are deliberately separate from the historical 50 ms / 5 s datasets.

Feature modes:
  raw_axis_216
  magnitude_72
  raw_plus_magnitude_288

Models:
  Logistic Regression
  RBF SVM
  Random Forest

Evaluation:
  - random stratified CV as a reference
  - leave-one-board-pose-out as the orientation robustness metric
"""

import argparse
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
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
SAMPLE_INTERVAL_MS = 40
WINDOW_SECONDS = 4.0
PROTOCOL_ID = "MPU6050_100X6_40MS_4S"
RANDOM_STATE = 42

EXPECTED_POSES = [
    "BOARD_FACE_UP",
    "BOARD_FACE_DOWN",
    "BOARD_FACE_VERTICAL",
]
FEATURE_MODES = [
    "raw_axis_216",
    "magnitude_72",
    "raw_plus_magnitude_288",
]

TASKS = {
    "numbers": {
        "task_id": "numbers_0_9",
        "classes": list("0123456789"),
        "data_dir": REPO_ROOT / "data/numbers/board_pose_study_40ms",
        "results_dir": REPO_ROOT / "results/numbers/board_pose_study_40ms",
        "model_file": REPO_ROOT / "models/mpu6050_numbers_40ms_board_pose_best_model.pkl",
        "prefix": "numbers_0_9_40ms",
    },
    "letters": {
        "task_id": "letters_A_Z",
        "classes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "data_dir": REPO_ROOT / "data/letters/board_pose_study_40ms",
        "results_dir": REPO_ROOT / "results/letters/board_pose_study_40ms",
        "model_file": REPO_ROOT / "models/mpu6050_letters_40ms_board_pose_best_model.pkl",
        "prefix": "letters_A_Z_40ms",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    return parser.parse_args()


def normalize_label(value):
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".0") else text


def get_sensor_columns(df):
    columns = [c for c in df.columns if c.startswith("value_")]

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


def extract_raw(recording):
    features = []
    for segment in np.array_split(recording, NUM_SEGMENTS, axis=0):
        for channel in range(NUM_CHANNELS):
            features.extend(segment_statistics(segment[:, channel]))
    result = np.asarray(features, dtype=float)
    if result.shape != (216,):
        raise RuntimeError(f"Expected 216 raw features, got {result.shape}")
    return result


def extract_magnitude(recording):
    magnitude_recording = np.column_stack([
        np.linalg.norm(recording[:, 0:3], axis=1),
        np.linalg.norm(recording[:, 3:6], axis=1),
    ])
    features = []
    for segment in np.array_split(magnitude_recording, NUM_SEGMENTS, axis=0):
        for channel in range(2):
            features.extend(segment_statistics(segment[:, channel]))
    result = np.asarray(features, dtype=float)
    if result.shape != (72,):
        raise RuntimeError(f"Expected 72 magnitude features, got {result.shape}")
    return result


def extract_features(recording, mode):
    raw = extract_raw(recording)
    magnitude = extract_magnitude(recording)
    if mode == "raw_axis_216":
        return raw
    if mode == "magnitude_72":
        return magnitude
    if mode == "raw_plus_magnitude_288":
        return np.concatenate([raw, magnitude])
    raise ValueError(mode)


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=10000,
                C=1.0,
                random_state=RANDOM_STATE,
            )),
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=1200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def score(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        )),
    }


def validate_protocol(df, path):
    required = [
        "actual",
        "orientation_id",
        "session_id",
        "participant_id",
        "protocol_id",
        "sample_interval_ms",
        "window_seconds",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing metadata {missing}")

    protocols = set(df["protocol_id"].astype(str).unique())
    if protocols != {PROTOCOL_ID}:
        raise RuntimeError(f"{path.name}: wrong protocol {protocols}")

    intervals = set(pd.to_numeric(df["sample_interval_ms"], errors="coerce").dropna())
    windows = set(pd.to_numeric(df["window_seconds"], errors="coerce").dropna())
    if intervals != {SAMPLE_INTERVAL_MS} or windows != {WINDOW_SECONDS}:
        raise RuntimeError(
            f"{path.name}: expected 40 ms / 4 s, got {intervals} / {windows}"
        )


def load_dataset(config):
    data_dir = config["data_dir"]
    files = sorted(data_dir.glob(f"{config['prefix']}_*.csv"))
    if not files:
        raise RuntimeError(f"No matching CSV files found in {data_dir}")

    frames = []
    reference_columns = None

    for path in files:
        df = pd.read_csv(path)
        validate_protocol(df, path)
        labels = df["actual"].map(normalize_label)
        unexpected = sorted(set(labels) - set(config["classes"]))
        if unexpected:
            raise RuntimeError(f"{path.name}: unexpected labels {unexpected}")

        columns = get_sensor_columns(df)
        if len(columns) != NUM_SAMPLES * NUM_CHANNELS:
            raise RuntimeError(f"{path.name}: expected 600 sensor columns")
        if reference_columns is not None and columns != reference_columns:
            raise RuntimeError(f"{path.name}: sensor-column order mismatch")
        reference_columns = columns

        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if int(numeric.isna().sum().sum()) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path.name}: invalid sensor values")

        copied = df.copy()
        copied["actual"] = labels
        copied["source_file"] = path.name
        frames.append(copied)
        print(
            f"Loaded {path.name}: {len(df)} rows, "
            f"pose={df['orientation_id'].iloc[0]}"
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    poses = sorted(combined["orientation_id"].astype(str).unique())
    missing_poses = sorted(set(EXPECTED_POSES) - set(poses))
    if missing_poses:
        raise RuntimeError(f"Missing required poses: {missing_poses}")

    sensor_columns = get_sensor_columns(combined)
    values = combined[sensor_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    duplicates = int(pd.DataFrame(values).duplicated().sum())
    if duplicates:
        raise RuntimeError(f"Exact duplicate recordings: {duplicates}")

    y = combined["actual"].map(normalize_label).to_numpy(dtype=str)
    groups = combined["orientation_id"].astype(str).to_numpy()
    sessions = combined["session_id"].astype(str).to_numpy()
    recordings = values.reshape(len(combined), NUM_SAMPLES, NUM_CHANNELS)

    print("Combined rows:", len(combined))
    print("Pose counts:")
    print(combined["orientation_id"].value_counts())
    print("Class counts:")
    print(pd.Series(y).value_counts().sort_index())
    return combined, recordings, y, groups, sessions


def main():
    args = parse_args()
    config = TASKS[args.task]
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    config["model_file"].parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"{config['task_id']} - 40 MS / 4 S BOARD-POSE TRAINING")
    print("=" * 72)

    df, recordings, y, groups, sessions = load_dataset(config)
    min_count = int(pd.Series(y).value_counts().min())
    if min_count < 2:
        raise RuntimeError("Not enough examples per class")
    random_cv = StratifiedKFold(
        n_splits=min(5, min_count),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grouped_cv = LeaveOneGroupOut()

    rows = []
    prediction_cache = {}
    model_cache = {}
    feature_cache = {}

    for mode in FEATURE_MODES:
        print("=" * 72)
        print("FEATURE MODE:", mode)
        print("=" * 72)
        X = np.vstack([extract_features(recording, mode) for recording in recordings])
        feature_cache[mode] = X
        print("Feature matrix:", X.shape)

        for name, model in build_models().items():
            random_pred = cross_val_predict(model, X, y, cv=random_cv, n_jobs=-1)
            grouped_pred = cross_val_predict(
                model,
                X,
                y,
                groups=groups,
                cv=grouped_cv,
                n_jobs=-1,
            )
            random_metrics = score(y, random_pred)
            grouped_metrics = score(y, grouped_pred)
            rows.append({
                "feature_mode": mode,
                "model": name,
                "random_accuracy": random_metrics["accuracy"],
                "random_balanced_accuracy": random_metrics["balanced_accuracy"],
                "random_macro_f1": random_metrics["macro_f1"],
                "grouped_accuracy": grouped_metrics["accuracy"],
                "grouped_balanced_accuracy": grouped_metrics["balanced_accuracy"],
                "grouped_macro_f1": grouped_metrics["macro_f1"],
            })
            print(
                f"{name}: random={random_metrics['accuracy']:.4f}, "
                f"leave-one-pose-out={grouped_metrics['accuracy']:.4f}"
            )
            prediction_cache[(mode, name)] = np.asarray(grouped_pred, dtype=str)
            model.fit(X, y)
            model_cache[(mode, name)] = model

    comparison = pd.DataFrame(rows).sort_values(
        [
            "grouped_balanced_accuracy",
            "grouped_macro_f1",
            "grouped_accuracy",
            "random_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_mode = str(comparison.iloc[0]["feature_mode"])
    selected_name = str(comparison.iloc[0]["model"])
    selected_model = model_cache[(selected_mode, selected_name)]
    selected_pred = prediction_cache[(selected_mode, selected_name)]
    selected_metrics = score(y, selected_pred)

    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False))
    print("=" * 72)
    print("SELECTED 40 MS / 4 S PIPELINE")
    print("=" * 72)
    print("Task:", config["task_id"])
    print("Feature mode:", selected_mode)
    print("Model:", selected_name)
    print(f"Grouped accuracy: {selected_metrics['accuracy']:.4f}")
    print(f"Grouped balanced accuracy: {selected_metrics['balanced_accuracy']:.4f}")
    print(f"Grouped macro F1: {selected_metrics['macro_f1']:.4f}")
    print(classification_report(
        y,
        selected_pred,
        labels=config["classes"],
        digits=4,
        zero_division=0,
    ))

    per_pose_rows = []
    for pose in EXPECTED_POSES:
        mask = groups == pose
        per_pose_rows.append({
            "held_out_pose": pose,
            "rows": int(mask.sum()),
            **score(y[mask], selected_pred[mask]),
        })
    per_pose = pd.DataFrame(per_pose_rows)
    print("PER-POSE RESULTS")
    print(per_pose.to_string(index=False))

    prefix = config["prefix"]
    results_dir = config["results_dir"]
    comparison_file = results_dir / f"{prefix}_model_comparison.csv"
    predictions_file = results_dir / f"{prefix}_grouped_predictions.csv"
    confusion_file = results_dir / f"{prefix}_grouped_confusion_matrix.csv"
    per_pose_file = results_dir / f"{prefix}_leave_one_pose_metrics.csv"
    summary_file = results_dir / f"{prefix}_training_summary.txt"

    comparison.to_csv(comparison_file, index=False)
    per_pose.to_csv(per_pose_file, index=False)

    predictions_df = pd.DataFrame({
        "actual": y,
        "prediction": selected_pred,
        "correct": y == selected_pred,
        "orientation_id": groups,
        "session_id": sessions,
    })
    for column in ("participant_id", "trial_number", "source_file"):
        if column in df.columns:
            predictions_df[column] = df[column].to_numpy()
    predictions_df.to_csv(predictions_file, index=False)

    pd.DataFrame(
        confusion_matrix(y, selected_pred, labels=config["classes"]),
        index=[f"actual_{label}" for label in config["classes"]],
        columns=[f"pred_{label}" for label in config["classes"]],
    ).to_csv(confusion_file)

    saved = {
        "model": selected_model,
        "model_name": selected_name,
        "feature_mode": selected_mode,
        "classes": config["classes"],
        "task_id": config["task_id"],
        "protocol_id": PROTOCOL_ID,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "sampling_rate_hz": 1000.0 / SAMPLE_INTERVAL_MS,
        "window_seconds": WINDOW_SECONDS,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "training_rows": len(df),
        "supported_poses": EXPECTED_POSES,
        "grouped_metrics": selected_metrics,
    }
    with open(config["model_file"], "wb") as handle:
        pickle.dump(saved, handle)

    summary_file.write_text(
        f"Task: {config['task_id']}\n"
        f"Protocol: {PROTOCOL_ID}\n"
        f"Rows: {len(df)}\n"
        f"Feature mode: {selected_mode}\n"
        f"Model: {selected_name}\n"
        f"Grouped accuracy: {selected_metrics['accuracy']:.6f}\n"
        f"Grouped balanced accuracy: {selected_metrics['balanced_accuracy']:.6f}\n"
        f"Grouped macro F1: {selected_metrics['macro_f1']:.6f}\n\n"
        + comparison.to_string(index=False)
        + "\n\n"
        + per_pose.to_string(index=False)
    )

    print("Saved model:", config["model_file"])
    print("Saved comparison:", comparison_file)
    print("Saved predictions:", predictions_file)
    print("Saved confusion matrix:", confusion_file)
    print("Saved per-pose metrics:", per_pose_file)
    print("Saved summary:", summary_file)


if __name__ == "__main__":
    main()
