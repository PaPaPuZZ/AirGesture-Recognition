#!/usr/bin/env python3
"""
TRAIN 40 MS / 4 S NUMERIC MODELS ACROSS MULTIPLE BOARD-POSE SESSIONS

Expected input directory:
    data/numbers/board_pose_study_40ms/

Expected controlled dataset for the first multisession comparison:
    3 board poses
    x 2 independent sessions per pose
    x 10 digits
    x 5 trials
    = 300 recordings

Compared strategies:
    shared
        One classifier is trained across all supported board poses.

    pose_routed
        One classifier is trained per board pose. The known/detected pose
        selects the appropriate classifier at inference time.

Compared feature modes:
    raw_axis_216
    magnitude_72
    raw_plus_magnitude_288

Compared classifiers:
    Logistic Regression
    RBF SVM
    Random Forest

Evaluation:
    random stratified CV
        Reference only; recordings from all sessions may mix across folds.

    leave-one-session-ID-out
        Holds out one complete collection session.

    bidirectional round holdout (PRIMARY)
        Train on all *_01 sessions and test on all *_02 sessions, then reverse.
        This measures independent-session generalization while keeping all
        three supported poses represented in training.

The selected final bundle can contain either one shared classifier or three
pose-routed classifiers. Fresh certification data are never loaded here.
"""

from __future__ import annotations

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
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "numbers" / "board_pose_study_40ms"
RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "numbers"
    / "board_pose_study_40ms"
    / "multisession"
)
MODELS_DIR = REPO_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "mpu6050_numbers_40ms_multisession_best_model.pkl"
COMPARISON_FILE = RESULTS_DIR / "numbers_40ms_multisession_model_comparison.csv"
ROUND_PREDICTIONS_FILE = RESULTS_DIR / "numbers_40ms_round_holdout_predictions.csv"
SESSION_PREDICTIONS_FILE = RESULTS_DIR / "numbers_40ms_session_holdout_predictions.csv"
ROUND_CONFUSION_FILE = RESULTS_DIR / "numbers_40ms_round_holdout_confusion_matrix.csv"
PER_POSE_FILE = RESULTS_DIR / "numbers_40ms_round_holdout_per_pose_metrics.csv"
PER_DIRECTION_FILE = RESULTS_DIR / "numbers_40ms_round_holdout_directions.csv"
SUMMARY_FILE = RESULTS_DIR / "numbers_40ms_multisession_training_summary.txt"

NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
SAMPLE_INTERVAL_MS = 40
WINDOW_SECONDS = 4.0
PROTOCOL_ID = "MPU6050_100X6_40MS_4S"
RANDOM_STATE = 42

CLASSES = list("0123456789")
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


def normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".0") else text


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("value_")]

    def key(column: str) -> tuple[int, int]:
        sample, channel = column.replace("value_", "").split("_")
        return int(sample), int(channel)

    return sorted(columns, key=key)


def parse_session_round(session_id: str) -> int:
    match = re.search(r"(?:_|-)(\d+)$", str(session_id).strip())
    if not match:
        raise RuntimeError(
            "Every session_id must end in a numeric round, such as _01 or _02. "
            f"Could not parse: {session_id!r}"
        )
    return int(match.group(1))


def segment_statistics(signal: np.ndarray) -> list[float]:
    signal = np.asarray(signal, dtype=float)
    delta = np.diff(signal)
    return [
        float(np.mean(signal)),
        float(np.std(signal)),
        float(np.min(signal)),
        float(np.max(signal)),
        float(np.ptp(signal)),
        float(np.sqrt(np.mean(signal**2))),
        float(np.mean(np.abs(delta))),
        float(np.max(np.abs(delta))),
        float(np.sum(np.abs(delta))),
    ]


def extract_raw(recording: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for segment in np.array_split(recording, NUM_SEGMENTS, axis=0):
        for channel in range(NUM_CHANNELS):
            features.extend(segment_statistics(segment[:, channel]))
    result = np.asarray(features, dtype=float)
    if result.shape != (216,):
        raise RuntimeError(f"Expected 216 raw-axis features, got {result.shape}")
    return result


def extract_magnitude(recording: np.ndarray) -> np.ndarray:
    magnitude_recording = np.column_stack(
        [
            np.linalg.norm(recording[:, 0:3], axis=1),
            np.linalg.norm(recording[:, 3:6], axis=1),
        ]
    )
    features: list[float] = []
    for segment in np.array_split(magnitude_recording, NUM_SEGMENTS, axis=0):
        for channel in range(2):
            features.extend(segment_statistics(segment[:, channel]))
    result = np.asarray(features, dtype=float)
    if result.shape != (72,):
        raise RuntimeError(f"Expected 72 magnitude features, got {result.shape}")
    return result


def extract_features(recording: np.ndarray, mode: str) -> np.ndarray:
    raw = extract_raw(recording)
    magnitude = extract_magnitude(recording)
    if mode == "raw_axis_216":
        return raw
    if mode == "magnitude_72":
        return magnitude
    if mode == "raw_plus_magnitude_288":
        return np.concatenate([raw, magnitude])
    raise ValueError(f"Unsupported feature mode: {mode}")


def validate_protocol(df: pd.DataFrame, path: Path) -> None:
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
        raise RuntimeError(f"{path.name}: missing metadata columns {missing}")

    protocols = set(df["protocol_id"].astype(str).unique())
    if protocols != {PROTOCOL_ID}:
        raise RuntimeError(f"{path.name}: expected protocol {PROTOCOL_ID}, got {protocols}")

    intervals = set(
        pd.to_numeric(df["sample_interval_ms"], errors="coerce").dropna().tolist()
    )
    windows = set(
        pd.to_numeric(df["window_seconds"], errors="coerce").dropna().tolist()
    )
    if intervals != {SAMPLE_INTERVAL_MS} or windows != {WINDOW_SECONDS}:
        raise RuntimeError(
            f"{path.name}: expected 40 ms / 4.0 s, got {intervals} / {windows}"
        )


def load_dataset():
    files = sorted(DATA_DIR.glob("numbers_0_9_40ms_*.csv"))
    if not files:
        raise RuntimeError(f"No 40 ms numeric CSV files found in {DATA_DIR}")

    frames: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None

    for path in files:
        df = pd.read_csv(path)
        validate_protocol(df, path)

        labels = df["actual"].map(normalize_label)
        unexpected = sorted(set(labels) - set(CLASSES))
        if unexpected:
            raise RuntimeError(f"{path.name}: unexpected labels {unexpected}")

        poses = set(df["orientation_id"].astype(str).unique())
        sessions = set(df["session_id"].astype(str).unique())
        if len(poses) != 1 or len(sessions) != 1:
            raise RuntimeError(
                f"{path.name}: each file must contain one pose and one session; "
                f"got poses={poses}, sessions={sessions}"
            )

        pose = next(iter(poses))
        session = next(iter(sessions))
        if pose not in EXPECTED_POSES:
            raise RuntimeError(f"{path.name}: unsupported pose {pose}")

        if len(df) != 50:
            raise RuntimeError(f"{path.name}: expected 50 rows, found {len(df)}")
        for digit in CLASSES:
            count = int((labels == digit).sum())
            if count != 5:
                raise RuntimeError(
                    f"{path.name}: digit {digit} expected 5 rows, found {count}"
                )

        columns = get_sensor_columns(df)
        if len(columns) != NUM_SAMPLES * NUM_CHANNELS:
            raise RuntimeError(
                f"{path.name}: expected 600 sensor columns, found {len(columns)}"
            )
        if reference_columns is not None and columns != reference_columns:
            raise RuntimeError(f"{path.name}: sensor-column order mismatch")
        reference_columns = columns

        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if int(numeric.isna().sum().sum()) != 0 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path.name}: invalid sensor values")

        copied = df.copy()
        copied["actual"] = labels
        copied["session_round"] = parse_session_round(session)
        copied["source_file"] = path.name
        frames.append(copied)

        print(
            f"Loaded {path.name}: rows={len(df)}, pose={pose}, "
            f"session={session}, round={copied['session_round'].iloc[0]}"
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sensor_columns = get_sensor_columns(combined)
    values = combined[sensor_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    duplicates = int(pd.DataFrame(values).duplicated().sum())
    if duplicates:
        raise RuntimeError(f"Exact duplicate recordings across files: {duplicates}")

    poses = sorted(combined["orientation_id"].astype(str).unique())
    missing_poses = sorted(set(EXPECTED_POSES) - set(poses))
    if missing_poses:
        raise RuntimeError(f"Missing poses: {missing_poses}")

    rounds = sorted(combined["session_round"].astype(int).unique())
    if rounds != [1, 2]:
        raise RuntimeError(
            "This first multisession comparison expects exactly rounds 01 and 02. "
            f"Found rounds: {rounds}"
        )

    session_table = (
        combined[["orientation_id", "session_id", "session_round"]]
        .drop_duplicates()
        .sort_values(["orientation_id", "session_round"])
    )
    for pose in EXPECTED_POSES:
        pose_sessions = session_table[session_table["orientation_id"] == pose]
        pose_rounds = sorted(pose_sessions["session_round"].astype(int).tolist())
        if pose_rounds != [1, 2]:
            raise RuntimeError(
                f"{pose}: expected one session in rounds 01 and 02, found {pose_rounds}"
            )

    y = combined["actual"].map(normalize_label).to_numpy(dtype=str)
    pose_groups = combined["orientation_id"].astype(str).to_numpy()
    session_groups = combined["session_id"].astype(str).to_numpy()
    round_groups = combined["session_round"].astype(int).to_numpy()
    recordings = values.reshape(len(combined), NUM_SAMPLES, NUM_CHANNELS)

    print()
    print("=" * 72)
    print("MULTISESSION DATASET VALIDATION")
    print("=" * 72)
    print("Rows:", len(combined))
    print("Sessions:", session_table.shape[0])
    print("Pose counts:")
    print(combined["orientation_id"].value_counts().sort_index())
    print("Round counts:")
    print(combined["session_round"].value_counts().sort_index())
    print("Class counts:")
    print(pd.Series(y, name="actual").value_counts().sort_index())
    print("Exact duplicates:", duplicates)

    if len(combined) != 300:
        raise RuntimeError(f"Expected 300 rows, found {len(combined)}")

    return combined, recordings, y, pose_groups, session_groups, round_groups


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=10000,
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RBF SVM": Pipeline(
            [
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
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=1500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
    }


def shared_session_predictions(model, X, y, session_groups):
    return np.asarray(
        cross_val_predict(
            model,
            X,
            y,
            groups=session_groups,
            cv=LeaveOneGroupOut(),
            n_jobs=-1,
        ),
        dtype=str,
    )


def shared_round_predictions(model, X, y, round_groups):
    predictions = np.empty(len(y), dtype=object)
    direction_rows = []

    for held_out_round in sorted(np.unique(round_groups)):
        train_mask = round_groups != held_out_round
        test_mask = round_groups == held_out_round
        fitted = clone(model).fit(X[train_mask], y[train_mask])
        fold_pred = np.asarray(fitted.predict(X[test_mask]), dtype=str)
        predictions[test_mask] = fold_pred
        direction_rows.append(
            {
                "strategy": "shared",
                "held_out_round": int(held_out_round),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                **score(y[test_mask], fold_pred),
            }
        )

    return np.asarray(predictions, dtype=str), direction_rows


def routed_random_predictions(model, X, y, pose_groups):
    predictions = np.empty(len(y), dtype=object)

    for pose in EXPECTED_POSES:
        mask = pose_groups == pose
        min_count = int(pd.Series(y[mask]).value_counts().min())
        cv = StratifiedKFold(
            n_splits=min(5, min_count),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        predictions[mask] = cross_val_predict(
            model,
            X[mask],
            y[mask],
            cv=cv,
            n_jobs=-1,
        )

    return np.asarray(predictions, dtype=str)


def routed_session_predictions(model, X, y, pose_groups, session_groups):
    predictions = np.empty(len(y), dtype=object)

    for held_out_session in sorted(np.unique(session_groups)):
        test_mask = session_groups == held_out_session
        session_poses = np.unique(pose_groups[test_mask])
        if len(session_poses) != 1:
            raise RuntimeError(
                f"Session {held_out_session} contains multiple poses: {session_poses}"
            )
        pose = str(session_poses[0])
        train_mask = (pose_groups == pose) & (session_groups != held_out_session)
        if not np.any(train_mask):
            raise RuntimeError(f"No same-pose training rows for {held_out_session}")
        fitted = clone(model).fit(X[train_mask], y[train_mask])
        predictions[test_mask] = fitted.predict(X[test_mask])

    return np.asarray(predictions, dtype=str)


def routed_round_predictions(model, X, y, pose_groups, round_groups):
    predictions = np.empty(len(y), dtype=object)
    direction_rows = []

    for held_out_round in sorted(np.unique(round_groups)):
        test_round_mask = round_groups == held_out_round

        for pose in EXPECTED_POSES:
            train_mask = (round_groups != held_out_round) & (pose_groups == pose)
            test_mask = (round_groups == held_out_round) & (pose_groups == pose)
            if not np.any(train_mask) or not np.any(test_mask):
                raise RuntimeError(
                    f"Missing train/test rows for pose={pose}, round={held_out_round}"
                )
            fitted = clone(model).fit(X[train_mask], y[train_mask])
            predictions[test_mask] = fitted.predict(X[test_mask])

        fold_pred = np.asarray(predictions[test_round_mask], dtype=str)
        direction_rows.append(
            {
                "strategy": "pose_routed",
                "held_out_round": int(held_out_round),
                "train_rows": int((round_groups != held_out_round).sum()),
                "test_rows": int(test_round_mask.sum()),
                **score(y[test_round_mask], fold_pred),
            }
        )

    return np.asarray(predictions, dtype=str), direction_rows


def fit_final_bundle(strategy, model, X, y, pose_groups):
    if strategy == "shared":
        return {"model": clone(model).fit(X, y)}

    pose_models = {}
    for pose in EXPECTED_POSES:
        mask = pose_groups == pose
        pose_models[pose] = clone(model).fit(X[mask], y[mask])
    return {"pose_models": pose_models}


def main() -> None:
    print("=" * 72)
    print("40 MS NUMERIC MULTISESSION + POSE-ROUTED TRAINING")
    print("=" * 72)

    df, recordings, y, pose_groups, session_groups, round_groups = load_dataset()

    min_count = int(pd.Series(y).value_counts().min())
    random_cv = StratifiedKFold(
        n_splits=min(5, min_count),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    prediction_cache = {}
    direction_cache = {}
    feature_cache = {}
    model_templates = build_models()

    for feature_mode in FEATURE_MODES:
        print()
        print("=" * 72)
        print("FEATURE MODE:", feature_mode)
        print("=" * 72)

        X = np.vstack(
            [extract_features(recording, feature_mode) for recording in recordings]
        )
        feature_cache[feature_mode] = X
        print("Feature matrix:", X.shape)

        for model_name, model in model_templates.items():
            print()
            print("-" * 72)
            print(model_name)
            print("-" * 72)

            shared_random = np.asarray(
                cross_val_predict(model, X, y, cv=random_cv, n_jobs=-1), dtype=str
            )
            shared_session = shared_session_predictions(
                model, X, y, session_groups
            )
            shared_round, shared_directions = shared_round_predictions(
                model, X, y, round_groups
            )

            routed_random = routed_random_predictions(model, X, y, pose_groups)
            routed_session = routed_session_predictions(
                model, X, y, pose_groups, session_groups
            )
            routed_round, routed_directions = routed_round_predictions(
                model, X, y, pose_groups, round_groups
            )

            for strategy, random_pred, session_pred, round_pred, directions in [
                (
                    "shared",
                    shared_random,
                    shared_session,
                    shared_round,
                    shared_directions,
                ),
                (
                    "pose_routed",
                    routed_random,
                    routed_session,
                    routed_round,
                    routed_directions,
                ),
            ]:
                random_metrics = score(y, random_pred)
                session_metrics = score(y, session_pred)
                round_metrics = score(y, round_pred)

                comparison_rows.append(
                    {
                        "strategy": strategy,
                        "feature_mode": feature_mode,
                        "model": model_name,
                        "random_accuracy": random_metrics["accuracy"],
                        "random_balanced_accuracy": random_metrics[
                            "balanced_accuracy"
                        ],
                        "random_macro_f1": random_metrics["macro_f1"],
                        "session_holdout_accuracy": session_metrics["accuracy"],
                        "session_holdout_balanced_accuracy": session_metrics[
                            "balanced_accuracy"
                        ],
                        "session_holdout_macro_f1": session_metrics["macro_f1"],
                        "round_holdout_accuracy": round_metrics["accuracy"],
                        "round_holdout_balanced_accuracy": round_metrics[
                            "balanced_accuracy"
                        ],
                        "round_holdout_macro_f1": round_metrics["macro_f1"],
                    }
                )
                key = (strategy, feature_mode, model_name)
                prediction_cache[("round", *key)] = round_pred
                prediction_cache[("session", *key)] = session_pred
                direction_cache[key] = directions

                print(
                    f"{strategy}: random={random_metrics['accuracy']:.4f}, "
                    f"session={session_metrics['accuracy']:.4f}, "
                    f"round={round_metrics['accuracy']:.4f}"
                )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        [
            "round_holdout_balanced_accuracy",
            "round_holdout_macro_f1",
            "round_holdout_accuracy",
            "session_holdout_balanced_accuracy",
            "random_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_strategy = str(comparison.iloc[0]["strategy"])
    selected_feature = str(comparison.iloc[0]["feature_mode"])
    selected_model_name = str(comparison.iloc[0]["model"])
    selected_key = (selected_strategy, selected_feature, selected_model_name)
    selected_round_pred = prediction_cache[("round", *selected_key)]
    selected_session_pred = prediction_cache[("session", *selected_key)]
    selected_round_metrics = score(y, selected_round_pred)
    selected_session_metrics = score(y, selected_session_pred)

    print()
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False))

    print()
    print("=" * 72)
    print("SELECTED 40 MS MULTISESSION PIPELINE")
    print("=" * 72)
    print("Strategy:", selected_strategy)
    print("Feature mode:", selected_feature)
    print("Model:", selected_model_name)
    print(f"Round-holdout accuracy: {selected_round_metrics['accuracy']:.4f}")
    print(
        "Round-holdout balanced accuracy:",
        f"{selected_round_metrics['balanced_accuracy']:.4f}",
    )
    print(f"Round-holdout macro F1: {selected_round_metrics['macro_f1']:.4f}")
    print(f"Session-holdout accuracy: {selected_session_metrics['accuracy']:.4f}")
    print()
    print("ROUND-HELD-OUT CLASSIFICATION REPORT")
    print(
        classification_report(
            y,
            selected_round_pred,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    per_pose_rows = []
    for pose in EXPECTED_POSES:
        mask = pose_groups == pose
        per_pose_rows.append(
            {
                "pose": pose,
                "rows": int(mask.sum()),
                **score(y[mask], selected_round_pred[mask]),
            }
        )
    per_pose = pd.DataFrame(per_pose_rows)
    print("PER-POSE ROUND-HOLDOUT RESULTS")
    print(per_pose.to_string(index=False))

    directions = pd.DataFrame(direction_cache[selected_key])
    print()
    print("BIDIRECTIONAL ROUND RESULTS")
    print(directions.to_string(index=False))

    comparison.to_csv(COMPARISON_FILE, index=False)
    per_pose.to_csv(PER_POSE_FILE, index=False)
    directions.to_csv(PER_DIRECTION_FILE, index=False)

    round_predictions_df = pd.DataFrame(
        {
            "actual": y,
            "prediction": selected_round_pred,
            "correct": y == selected_round_pred,
            "orientation_id": pose_groups,
            "session_id": session_groups,
            "session_round": round_groups,
        }
    )
    session_predictions_df = round_predictions_df.copy()
    session_predictions_df["prediction"] = selected_session_pred
    session_predictions_df["correct"] = y == selected_session_pred

    for column in [
        "participant_id",
        "trial_number",
        "dominant_hand",
        "source_file",
    ]:
        if column in df.columns:
            round_predictions_df[column] = df[column].to_numpy()
            session_predictions_df[column] = df[column].to_numpy()

    round_predictions_df.to_csv(ROUND_PREDICTIONS_FILE, index=False)
    session_predictions_df.to_csv(SESSION_PREDICTIONS_FILE, index=False)

    pd.DataFrame(
        confusion_matrix(y, selected_round_pred, labels=CLASSES),
        index=[f"actual_{label}" for label in CLASSES],
        columns=[f"pred_{label}" for label in CLASSES],
    ).to_csv(ROUND_CONFUSION_FILE)

    selected_X = feature_cache[selected_feature]
    selected_template = build_models()[selected_model_name]
    fitted_components = fit_final_bundle(
        selected_strategy,
        selected_template,
        selected_X,
        y,
        pose_groups,
    )

    saved = {
        **fitted_components,
        "strategy": selected_strategy,
        "model_name": selected_model_name,
        "feature_mode": selected_feature,
        "classes": CLASSES,
        "poses": EXPECTED_POSES,
        "protocol_id": PROTOCOL_ID,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "sampling_rate_hz": 1000.0 / SAMPLE_INTERVAL_MS,
        "window_seconds": WINDOW_SECONDS,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "num_segments": NUM_SEGMENTS,
        "training_rows": int(len(df)),
        "training_sessions": sorted(np.unique(session_groups).tolist()),
        "training_rounds": sorted(np.unique(round_groups).astype(int).tolist()),
        "round_holdout_metrics": selected_round_metrics,
        "session_holdout_metrics": selected_session_metrics,
    }

    with open(MODEL_FILE, "wb") as handle:
        pickle.dump(saved, handle)

    summary_lines = [
        "40 MS NUMERIC MULTISESSION TRAINING SUMMARY\n",
        "=" * 70 + "\n",
        f"Rows: {len(df)}\n",
        f"Sessions: {len(np.unique(session_groups))}\n",
        f"Strategy: {selected_strategy}\n",
        f"Feature mode: {selected_feature}\n",
        f"Model: {selected_model_name}\n",
        f"Round-holdout accuracy: {selected_round_metrics['accuracy']:.4f}\n",
        f"Round-holdout balanced accuracy: "
        f"{selected_round_metrics['balanced_accuracy']:.4f}\n",
        f"Round-holdout macro F1: {selected_round_metrics['macro_f1']:.4f}\n",
        f"Session-holdout accuracy: {selected_session_metrics['accuracy']:.4f}\n\n",
        "FINAL COMPARISON\n",
        comparison.to_string(index=False),
        "\n\nPER-POSE ROUND-HOLDOUT RESULTS\n",
        per_pose.to_string(index=False),
        "\n\nBIDIRECTIONAL ROUND RESULTS\n",
        directions.to_string(index=False),
        "\n\nROUND-HELD-OUT CLASSIFICATION REPORT\n",
        classification_report(
            y,
            selected_round_pred,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        ),
    ]
    SUMMARY_FILE.write_text("".join(summary_lines))

    print()
    print("Saved model:", MODEL_FILE)
    print("Saved comparison:", COMPARISON_FILE)
    print("Saved round predictions:", ROUND_PREDICTIONS_FILE)
    print("Saved session predictions:", SESSION_PREDICTIONS_FILE)
    print("Saved confusion matrix:", ROUND_CONFUSION_FILE)
    print("Saved per-pose metrics:", PER_POSE_FILE)
    print("Saved round directions:", PER_DIRECTION_FILE)
    print("Saved summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
