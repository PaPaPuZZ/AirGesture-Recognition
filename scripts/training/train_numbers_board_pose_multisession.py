#!/usr/bin/env python3
"""
TRAIN NUMBER RECOGNITION ACROSS MULTIPLE BOARD-POSE SESSIONS

Expected data:
    3 supported board poses
    x 2 independent sessions per pose
    x 10 digits
    x 5 trials per digit
    = 300 recordings

Supported poses:
    BOARD_FACE_UP
    BOARD_FACE_DOWN
    BOARD_FACE_VERTICAL

Session 01 inputs:
    data/numbers/orientation_study/normalized/*.csv

Session 02 inputs:
    data/numbers/orientation_study/*.csv
    with orientation_id already set to one of the supported poses.

Feature modes:
    raw_axis_216
    magnitude_72
    raw_plus_magnitude_288

Candidate models:
    Logistic Regression
    RBF SVM
    Random Forest

Evaluation:
    1. Random stratified CV (reference only)
    2. Leave-one-session-ID-out GroupKFold
    3. Bidirectional session-round holdout:
       train round 01 -> test round 02
       train round 02 -> test round 01
    4. Leave-one-board-pose-out (secondary diagnostic)

Model selection prioritizes bidirectional session-round balanced accuracy,
because all supported board poses are represented in both training and test.
"""

import pickle
import re
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
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "numbers" / "orientation_study"
NORMALIZED_DIR = DATA_DIR / "normalized"
RESULTS_DIR = REPO_ROOT / "results" / "numbers" / "orientation_study" / "multisession"
MODELS_DIR = REPO_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "mpu6050_numbers_board_pose_multisession_best_model.pkl"
COMPARISON_FILE = RESULTS_DIR / "board_pose_multisession_model_comparison.csv"
SESSION_OOF_FILE = RESULTS_DIR / "board_pose_session_group_oof_predictions.csv"
ROUND_OOF_FILE = RESULTS_DIR / "board_pose_round_holdout_predictions.csv"
POSE_OOF_FILE = RESULTS_DIR / "board_pose_pose_holdout_predictions.csv"
ROUND_METRICS_FILE = RESULTS_DIR / "board_pose_round_holdout_metrics.csv"
SESSION_METRICS_FILE = RESULTS_DIR / "board_pose_session_group_metrics.csv"
CONFUSION_FILE = RESULTS_DIR / "board_pose_round_holdout_confusion_matrix.csv"
SUMMARY_FILE = RESULTS_DIR / "board_pose_multisession_training_summary.txt"

NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
DIGITS = [str(i) for i in range(10)]
SUPPORTED_POSES = [
    "BOARD_FACE_UP",
    "BOARD_FACE_DOWN",
    "BOARD_FACE_VERTICAL",
]
EXPECTED_PER_DIGIT_PER_SESSION = 5
EXPECTED_SESSIONS_PER_POSE = 2
RANDOM_STATE = 42

FEATURE_DIMS = {
    "raw_axis_216": 216,
    "magnitude_72": 72,
    "raw_plus_magnitude_288": 288,
}

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

    return np.asarray(features, dtype=float)


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

    return np.asarray(features, dtype=float)


def extract_features(recording, mode):
    raw = extract_raw_axis_features(recording)
    magnitude = extract_magnitude_features(recording)

    if mode == "raw_axis_216":
        result = raw
    elif mode == "magnitude_72":
        result = magnitude
    elif mode == "raw_plus_magnitude_288":
        result = np.concatenate([raw, magnitude])
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    expected = FEATURE_DIMS[mode]

    if result.shape != (expected,):
        raise RuntimeError(
            f"Feature mode {mode}: expected {expected}, got {result.shape}"
        )

    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"Feature mode {mode}: non-finite features")

    return result


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


def infer_round(session_id):
    value = str(session_id).upper()

    match = re.search(r"(?:_|-)(0?1|0?2)$", value)

    if not match:
        raise RuntimeError(
            f"Could not infer collection round from session_id: {session_id}. "
            "Use session IDs ending in _01 or _02."
        )

    return f"ROUND_{int(match.group(1)):02d}"


def validate_frame(df, source_name):
    required = [
        "actual",
        "orientation_id",
        "session_id",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"{source_name}: missing columns {missing}")

    labels = df["actual"].map(normalize_label)
    unexpected_labels = sorted(set(labels) - set(DIGITS))

    if unexpected_labels:
        raise RuntimeError(
            f"{source_name}: unexpected digit labels {unexpected_labels}"
        )

    poses = df["orientation_id"].astype(str).str.upper().unique().tolist()

    if len(poses) != 1 or poses[0] not in SUPPORTED_POSES:
        raise RuntimeError(
            f"{source_name}: expected one supported pose, found {poses}"
        )

    sessions = df["session_id"].astype(str).unique().tolist()

    if len(sessions) != 1:
        raise RuntimeError(
            f"{source_name}: expected one session_id, found {sessions}"
        )

    for digit in DIGITS:
        count = int((labels == digit).sum())

        if count != EXPECTED_PER_DIGIT_PER_SESSION:
            raise RuntimeError(
                f"{source_name}: digit {digit} expected "
                f"{EXPECTED_PER_DIGIT_PER_SESSION}, found {count}"
            )

    sensor_columns = get_sensor_columns(df)

    if len(sensor_columns) != NUM_SAMPLES * NUM_CHANNELS:
        raise RuntimeError(
            f"{source_name}: expected 600 sensor columns, "
            f"found {len(sensor_columns)}"
        )

    numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")

    if int(numeric.isna().sum().sum()) != 0:
        raise RuntimeError(f"{source_name}: missing/non-numeric sensor values")

    values = numeric.to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{source_name}: non-finite sensor values")

    if int(pd.DataFrame(values).duplicated().sum()) != 0:
        raise RuntimeError(f"{source_name}: exact duplicate recordings")

    validated = df.copy()
    validated["actual"] = labels
    validated["orientation_id"] = poses[0]
    validated["session_id"] = str(sessions[0])
    validated["session_round"] = infer_round(sessions[0])
    validated["source_file"] = source_name

    return validated, sensor_columns


def load_dataset():
    if not NORMALIZED_DIR.exists():
        raise FileNotFoundError(
            f"Missing normalized first-session directory: {NORMALIZED_DIR}"
        )

    candidate_files = list(NORMALIZED_DIR.glob("*.csv"))
    candidate_files.extend(DATA_DIR.glob("*.csv"))

    frames = []
    reference_sensor_columns = None
    seen_sessions = set()

    for path in sorted(set(candidate_files)):
        df = pd.read_csv(path)

        if "orientation_id" not in df.columns:
            continue

        pose_values = (
            df["orientation_id"]
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        if len(pose_values) != 1 or pose_values[0] not in SUPPORTED_POSES:
            continue

        validated, sensor_columns = validate_frame(df, path.name)
        session_id = validated["session_id"].iloc[0]

        if session_id in seen_sessions:
            raise RuntimeError(
                f"Duplicate session_id encountered: {session_id}"
            )

        seen_sessions.add(session_id)

        if reference_sensor_columns is None:
            reference_sensor_columns = sensor_columns
        elif list(sensor_columns) != list(reference_sensor_columns):
            raise RuntimeError(
                f"{path.name}: sensor-column ordering mismatch"
            )

        frames.append(validated)

        print(
            f"Loaded {path.name}: rows={len(validated)}, "
            f"pose={validated['orientation_id'].iloc[0]}, "
            f"session={session_id}, "
            f"round={validated['session_round'].iloc[0]}"
        )

    if not frames:
        raise RuntimeError("No supported board-pose session files found")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    session_table = (
        combined[["orientation_id", "session_id", "session_round"]]
        .drop_duplicates()
        .sort_values(["orientation_id", "session_round"])
    )

    print()
    print("Session table:")
    print(session_table.to_string(index=False))

    for pose in SUPPORTED_POSES:
        subset = session_table[session_table["orientation_id"] == pose]

        if len(subset) != EXPECTED_SESSIONS_PER_POSE:
            raise RuntimeError(
                f"Pose {pose}: expected {EXPECTED_SESSIONS_PER_POSE} sessions, "
                f"found {len(subset)}"
            )

        rounds = set(subset["session_round"])

        if rounds != {"ROUND_01", "ROUND_02"}:
            raise RuntimeError(
                f"Pose {pose}: expected ROUND_01 and ROUND_02, found {rounds}"
            )

    values = (
        combined[reference_sensor_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    duplicate_count = int(pd.DataFrame(values).duplicated().sum())

    if duplicate_count:
        raise RuntimeError(
            f"Combined dataset contains {duplicate_count} exact duplicates"
        )

    recordings = values.reshape(
        len(combined),
        NUM_SAMPLES,
        NUM_CHANNELS,
    )

    return combined, recordings


def grouped_predictions(model, X, y, groups):
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)

    if n_splits < 2:
        raise RuntimeError("Need at least two groups")

    cv = GroupKFold(n_splits=n_splits)

    return cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=-1,
    )


def bidirectional_round_predictions(model, X, y, rounds):
    predictions = np.empty(len(y), dtype=object)
    metric_rows = []

    for test_round in ["ROUND_01", "ROUND_02"]:
        train_mask = rounds != test_round
        test_mask = rounds == test_round

        fitted = clone(model)
        fitted.fit(X[train_mask], y[train_mask])
        fold_predictions = fitted.predict(X[test_mask])
        predictions[test_mask] = fold_predictions

        scores = score_predictions(y[test_mask], fold_predictions)
        metric_rows.append({
            "test_round": test_round,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            **scores,
        })

    return np.asarray(predictions, dtype=str), metric_rows


def main():
    print("=" * 78)
    print("BOARD-POSE MULTISESSION NUMBER TRAINING")
    print("=" * 78)

    df, recordings = load_dataset()

    y = df["actual"].to_numpy(dtype=str)
    sessions = df["session_id"].to_numpy(dtype=str)
    rounds = df["session_round"].to_numpy(dtype=str)
    poses = df["orientation_id"].to_numpy(dtype=str)

    print()
    print("Combined rows:", len(df))
    print("Pose counts:")
    print(df["orientation_id"].value_counts())
    print("Session counts:")
    print(df["session_id"].value_counts())
    print("Class counts:")
    print(df["actual"].value_counts().sort_index())

    comparison_rows = []
    artifacts = {}

    random_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for feature_mode in FEATURE_DIMS:
        print()
        print("=" * 78)
        print("FEATURE MODE:", feature_mode)
        print("=" * 78)

        X = np.vstack([
            extract_features(recording, feature_mode)
            for recording in recordings
        ])

        print("Feature matrix:", X.shape)

        for model_name, model in build_models().items():
            print()
            print("-" * 78)
            print(model_name)
            print("-" * 78)

            random_predictions = cross_val_predict(
                model,
                X,
                y,
                cv=random_cv,
                n_jobs=-1,
            )
            random_scores = score_predictions(y, random_predictions)

            session_predictions = grouped_predictions(
                model,
                X,
                y,
                sessions,
            )
            session_scores = score_predictions(y, session_predictions)

            round_predictions, round_metric_rows = bidirectional_round_predictions(
                model,
                X,
                y,
                rounds,
            )
            round_scores = score_predictions(y, round_predictions)

            pose_predictions = grouped_predictions(
                model,
                X,
                y,
                poses,
            )
            pose_scores = score_predictions(y, pose_predictions)

            print(
                "Random CV: "
                f"accuracy={random_scores['accuracy']:.4f} "
                f"balanced={random_scores['balanced_accuracy']:.4f} "
                f"macro_f1={random_scores['macro_f1']:.4f}"
            )
            print(
                "Leave-one-session-ID-out: "
                f"accuracy={session_scores['accuracy']:.4f} "
                f"balanced={session_scores['balanced_accuracy']:.4f} "
                f"macro_f1={session_scores['macro_f1']:.4f}"
            )
            print(
                "Bidirectional round holdout: "
                f"accuracy={round_scores['accuracy']:.4f} "
                f"balanced={round_scores['balanced_accuracy']:.4f} "
                f"macro_f1={round_scores['macro_f1']:.4f}"
            )
            print(
                "Leave-one-pose-out: "
                f"accuracy={pose_scores['accuracy']:.4f} "
                f"balanced={pose_scores['balanced_accuracy']:.4f} "
                f"macro_f1={pose_scores['macro_f1']:.4f}"
            )

            row = {
                "feature_mode": feature_mode,
                "model": model_name,
                "random_accuracy": random_scores["accuracy"],
                "random_balanced_accuracy": random_scores["balanced_accuracy"],
                "random_macro_f1": random_scores["macro_f1"],
                "session_group_accuracy": session_scores["accuracy"],
                "session_group_balanced_accuracy": session_scores["balanced_accuracy"],
                "session_group_macro_f1": session_scores["macro_f1"],
                "round_holdout_accuracy": round_scores["accuracy"],
                "round_holdout_balanced_accuracy": round_scores["balanced_accuracy"],
                "round_holdout_macro_f1": round_scores["macro_f1"],
                "pose_holdout_accuracy": pose_scores["accuracy"],
                "pose_holdout_balanced_accuracy": pose_scores["balanced_accuracy"],
                "pose_holdout_macro_f1": pose_scores["macro_f1"],
            }
            comparison_rows.append(row)

            artifacts[(feature_mode, model_name)] = {
                "X": X,
                "model": model,
                "session_predictions": np.asarray(session_predictions, dtype=str),
                "round_predictions": np.asarray(round_predictions, dtype=str),
                "pose_predictions": np.asarray(pose_predictions, dtype=str),
                "round_metric_rows": round_metric_rows,
            }

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        [
            "round_holdout_balanced_accuracy",
            "round_holdout_macro_f1",
            "round_holdout_accuracy",
            "session_group_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print()
    print("=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(comparison.to_string(index=False))

    best_feature_mode = str(comparison.iloc[0]["feature_mode"])
    best_model_name = str(comparison.iloc[0]["model"])
    best_artifact = artifacts[(best_feature_mode, best_model_name)]
    best_X = best_artifact["X"]
    best_model = clone(best_artifact["model"])
    best_round_predictions = best_artifact["round_predictions"]
    best_session_predictions = best_artifact["session_predictions"]
    best_pose_predictions = best_artifact["pose_predictions"]

    best_model.fit(best_X, y)

    round_scores = score_predictions(y, best_round_predictions)
    session_scores = score_predictions(y, best_session_predictions)
    pose_scores = score_predictions(y, best_pose_predictions)

    print()
    print("=" * 78)
    print("SELECTED BOARD-POSE MULTISESSION PIPELINE")
    print("=" * 78)
    print("Feature mode:", best_feature_mode)
    print("Model:", best_model_name)
    print(
        "Round-holdout: "
        f"accuracy={round_scores['accuracy']:.4f} "
        f"balanced={round_scores['balanced_accuracy']:.4f} "
        f"macro_f1={round_scores['macro_f1']:.4f}"
    )
    print(
        "Session-group: "
        f"accuracy={session_scores['accuracy']:.4f} "
        f"balanced={session_scores['balanced_accuracy']:.4f} "
        f"macro_f1={session_scores['macro_f1']:.4f}"
    )
    print(
        "Pose-holdout: "
        f"accuracy={pose_scores['accuracy']:.4f} "
        f"balanced={pose_scores['balanced_accuracy']:.4f} "
        f"macro_f1={pose_scores['macro_f1']:.4f}"
    )

    print()
    print("Round-held-out classification report:")
    print(
        classification_report(
            y,
            best_round_predictions,
            labels=DIGITS,
            digits=4,
            zero_division=0,
        )
    )

    comparison.to_csv(COMPARISON_FILE, index=False)

    prediction_base = pd.DataFrame({
        "actual": y,
        "orientation_id": poses,
        "session_id": sessions,
        "session_round": rounds,
    })

    session_oof = prediction_base.copy()
    session_oof["prediction"] = best_session_predictions
    session_oof["correct"] = y == best_session_predictions
    session_oof.to_csv(SESSION_OOF_FILE, index=False)

    round_oof = prediction_base.copy()
    round_oof["prediction"] = best_round_predictions
    round_oof["correct"] = y == best_round_predictions
    round_oof.to_csv(ROUND_OOF_FILE, index=False)

    pose_oof = prediction_base.copy()
    pose_oof["prediction"] = best_pose_predictions
    pose_oof["correct"] = y == best_pose_predictions
    pose_oof.to_csv(POSE_OOF_FILE, index=False)

    pd.DataFrame(best_artifact["round_metric_rows"]).to_csv(
        ROUND_METRICS_FILE,
        index=False,
    )

    session_metric_rows = []
    for session_id in sorted(np.unique(sessions)):
        mask = sessions == session_id
        scores = score_predictions(y[mask], best_session_predictions[mask])
        session_metric_rows.append({
            "session_id": session_id,
            "orientation_id": poses[mask][0],
            "session_round": rounds[mask][0],
            "rows": int(mask.sum()),
            **scores,
        })

    pd.DataFrame(session_metric_rows).to_csv(
        SESSION_METRICS_FILE,
        index=False,
    )

    pd.DataFrame(
        confusion_matrix(y, best_round_predictions, labels=DIGITS),
        index=[f"actual_{digit}" for digit in DIGITS],
        columns=[f"pred_{digit}" for digit in DIGITS],
    ).to_csv(CONFUSION_FILE)

    saved = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_mode": best_feature_mode,
        "classes": DIGITS,
        "supported_poses": SUPPORTED_POSES,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "num_segments": NUM_SEGMENTS,
        "expected_features": FEATURE_DIMS[best_feature_mode],
        "training_rows": int(len(df)),
        "session_ids": sorted(np.unique(sessions).tolist()),
        "session_rounds": sorted(np.unique(rounds).tolist()),
        "round_holdout_metrics": round_scores,
        "session_group_metrics": session_scores,
        "pose_holdout_metrics": pose_scores,
    }

    with open(MODEL_FILE, "wb") as handle:
        pickle.dump(saved, handle)

    summary_lines = [
        "BOARD-POSE MULTISESSION TRAINING SUMMARY\n",
        "=" * 72 + "\n",
        f"Rows: {len(df)}\n",
        f"Feature mode: {best_feature_mode}\n",
        f"Model: {best_model_name}\n",
        f"Round holdout: {round_scores}\n",
        f"Session group: {session_scores}\n",
        f"Pose holdout: {pose_scores}\n\n",
        "FINAL COMPARISON\n",
        comparison.to_string(index=False),
        "\n\nROUND-HELD-OUT CLASSIFICATION REPORT\n",
        classification_report(
            y,
            best_round_predictions,
            labels=DIGITS,
            digits=4,
            zero_division=0,
        ),
    ]
    SUMMARY_FILE.write_text("".join(summary_lines))

    print()
    print("Saved model:", MODEL_FILE)
    print("Saved comparison:", COMPARISON_FILE)
    print("Saved session OOF:", SESSION_OOF_FILE)
    print("Saved round OOF:", ROUND_OOF_FILE)
    print("Saved pose OOF:", POSE_OOF_FILE)
    print("Saved round metrics:", ROUND_METRICS_FILE)
    print("Saved session metrics:", SESSION_METRICS_FILE)
    print("Saved confusion matrix:", CONFUSION_FILE)
    print("Saved summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
