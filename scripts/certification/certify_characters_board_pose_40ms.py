#!/usr/bin/env python3
"""
FRESH LIVE CERTIFICATION FOR THE 40 MS / 4 S CHARACTER MODELS

Usage:
  python scripts/certification/certify_characters_board_pose_40ms.py --task numbers
  python scripts/certification/certify_characters_board_pose_40ms.py --task letters

Required firmware:
  arduino/mpu6050_capture_40ms_4s.ino

The certification recordings are saved as test-only evidence and are never
merged automatically into training data.
"""

import argparse
import glob
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial

REPO_ROOT = Path(__file__).resolve().parents[2]
BAUD_RATE = 115200
NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
SAMPLE_INTERVAL_MS = 40
WINDOW_SECONDS = 4.0
PROTOCOL_ID = "MPU6050_100X6_40MS_4S"

POSES = {
    "1": "BOARD_FACE_UP",
    "2": "BOARD_FACE_DOWN",
    "3": "BOARD_FACE_VERTICAL",
}

TASKS = {
    "numbers": {
        "task_id": "numbers_0_9",
        "classes": list("0123456789"),
        "model_file": REPO_ROOT / "models/mpu6050_numbers_40ms_board_pose_best_model.pkl",
        "results_dir": REPO_ROOT / "results/numbers/board_pose_study_40ms/certification",
    },
    "letters": {
        "task_id": "letters_A_Z",
        "classes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "model_file": REPO_ROOT / "models/mpu6050_letters_40ms_board_pose_best_model.pkl",
        "results_dir": REPO_ROOT / "results/letters/board_pose_study_40ms/certification",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    return args


def segment_statistics(signal):
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
    return np.asarray(features, dtype=float)


def extract_magnitude(recording):
    magnitude_recording = np.column_stack([
        np.linalg.norm(recording[:, :3], axis=1),
        np.linalg.norm(recording[:, 3:], axis=1),
    ])
    features = []
    for segment in np.array_split(magnitude_recording, NUM_SEGMENTS, axis=0):
        for channel in range(2):
            features.extend(segment_statistics(segment[:, channel]))
    return np.asarray(features, dtype=float)


def extract_features(recording, mode):
    raw = extract_raw(recording)
    magnitude = extract_magnitude(recording)
    if mode == "raw_axis_216":
        return raw
    if mode == "magnitude_72":
        return magnitude
    if mode == "raw_plus_magnitude_288":
        return np.concatenate([raw, magnitude])
    raise ValueError(f"Unsupported feature mode: {mode}")


def choose_pose():
    print("Board pose:")
    for key, pose in POSES.items():
        print(f"  {key}. {pose}")
    while True:
        choice = input("Choose [1/2/3]: ").strip()
        if choice in POSES:
            return POSES[choice]
        print("Invalid selection.")


def find_serial_ports():
    ports = []
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*"):
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def choose_serial_port():
    ports = find_serial_ports()
    if not ports:
        raise RuntimeError("No Arduino serial port found.")
    if len(ports) == 1:
        return ports[0]
    for index, port in enumerate(ports, start=1):
        print(f"{index}. {port}")
    while True:
        try:
            selected = int(input("Choose port: ").strip()) - 1
            if 0 <= selected < len(ports):
                return ports[selected]
        except ValueError:
            pass


def open_serial(port):
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    return ser


def reconnect_serial(old_ser=None):
    if old_ser is not None:
        try:
            old_ser.close()
        except Exception:
            pass
    time.sleep(2)
    ser = open_serial(choose_serial_port())
    print("Serial reconnected.")
    return ser


def countdown():
    print("Get ready...")
    for number in (3, 2, 1):
        print(number)
        time.sleep(1)
    print("GO - Recording now!")


def capture_recording(ser):
    ser.reset_input_buffer()
    ser.write(b"R")
    ser.flush()

    start_deadline = time.monotonic() + 8
    while time.monotonic() < start_deadline:
        raw = ser.readline()
        if raw and raw.decode("utf-8", errors="ignore").strip() == "START":
            break
    else:
        raise TimeoutError("Arduino did not send START.")

    samples = []
    started = time.monotonic()
    saw_end = False
    while time.monotonic() - started < 10:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line == "END":
            saw_end = True
            break
        parts = line.split(",")
        if len(parts) != NUM_CHANNELS:
            continue
        try:
            row = [float(value) for value in parts]
        except ValueError:
            continue
        if np.all(np.isfinite(row)):
            samples.append(row)

    elapsed = time.monotonic() - started
    if not saw_end:
        raise TimeoutError("Timed out before END.")
    recording = np.asarray(samples, dtype=float)
    if recording.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise RuntimeError(f"Expected (100, 6), got {recording.shape}")
    return recording, elapsed


def raw_row(actual, trial, pose, recording, elapsed):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actual": actual,
        "trial_number": trial,
        "orientation_id": pose,
        "protocol_id": PROTOCOL_ID,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "window_seconds": WINDOW_SECONDS,
        "capture_seconds": elapsed,
    }
    for sample_index in range(NUM_SAMPLES):
        for channel in range(NUM_CHANNELS):
            row[f"value_{sample_index}_{channel}"] = float(
                recording[sample_index, channel]
            )
    return row


def main():
    args = parse_args()
    config = TASKS[args.task]
    config["results_dir"].mkdir(parents=True, exist_ok=True)

    if not config["model_file"].exists():
        raise FileNotFoundError(config["model_file"])
    with open(config["model_file"], "rb") as handle:
        saved = pickle.load(handle)

    if saved.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Model was not trained on the 40 ms / 4 s protocol.")
    model = saved["model"]
    mode = saved["feature_mode"]
    classes = [str(value) for value in saved["classes"]]
    if classes != config["classes"]:
        raise RuntimeError("Saved classes do not match requested task.")

    pose = choose_pose()
    print("Model:", saved.get("model_name"))
    print("Feature mode:", mode)
    print("Pose:", pose)
    print("Trials per class:", args.trials)

    ser = open_serial(choose_serial_port())
    raw_rows = []
    result_rows = []

    try:
        for actual in classes:
            for trial in range(1, args.trials + 1):
                while True:
                    print(f"{actual} - Trial {trial}/{args.trials}")
                    answer = input("Press ENTER when ready or q to quit: ").strip().lower()
                    if answer == "q":
                        raise KeyboardInterrupt
                    countdown()
                    try:
                        recording, elapsed = capture_recording(ser)
                    except Exception as exc:
                        print("CAPTURE FAILED:", exc)
                        action = input(
                            "Press ENTER to retry, r to reconnect, or q to quit: "
                        ).strip().lower()
                        if action == "q":
                            raise KeyboardInterrupt
                        if action == "r":
                            ser = reconnect_serial(ser)
                        continue

                    features = extract_features(recording, mode).reshape(1, -1)
                    prediction = str(model.predict(features)[0])
                    probability_map = {}
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        probability_map = {
                            str(label): float(probability)
                            for label, probability in zip(model.classes_, probabilities)
                        }
                    confidence = probability_map.get(prediction, np.nan)
                    correct = prediction == actual
                    print("Prediction:", prediction)
                    print("Result:", "CORRECT" if correct else "WRONG")
                    if np.isfinite(confidence):
                        print(f"Confidence: {confidence * 100:.2f}%")

                    raw_rows.append(raw_row(actual, trial, pose, recording, elapsed))
                    result = {
                        "actual": actual,
                        "trial_number": trial,
                        "orientation_id": pose,
                        "prediction": prediction,
                        "correct": correct,
                        "confidence": confidence,
                        "capture_seconds": elapsed,
                    }
                    for label in classes:
                        result[f"prob_{label}"] = probability_map.get(label, np.nan)
                    result_rows.append(result)
                    break
    except KeyboardInterrupt:
        print("Certification stopped.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

    if not result_rows:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{config['task_id']}_40ms_{pose}_{timestamp}"
    raw_file = config["results_dir"] / f"{prefix}_raw.csv"
    results_file = config["results_dir"] / f"{prefix}_results.csv"
    summary_file = config["results_dir"] / f"{prefix}_summary.txt"

    raw_df = pd.DataFrame(raw_rows)
    results_df = pd.DataFrame(result_rows)
    raw_df.to_csv(raw_file, index=False)
    results_df.to_csv(results_file, index=False)

    correct_total = int(results_df["correct"].sum())
    total = len(results_df)
    lines = [
        f"Task: {config['task_id']}\n",
        f"Protocol: {PROTOCOL_ID}\n",
        f"Pose: {pose}\n",
        f"Overall: {correct_total}/{total}\n",
        f"Accuracy: {correct_total / total:.4f}\n\n",
    ]
    print("=" * 72)
    print("FINAL CERTIFICATION RESULTS")
    print("=" * 72)
    print(f"Overall: {correct_total}/{total}")
    print(f"Accuracy: {100 * correct_total / total:.2f}%")

    for label in classes:
        subset = results_df[results_df["actual"] == label]
        correct = int(subset["correct"].sum())
        line = f"{label}: {correct}/{len(subset)}"
        print(line)
        lines.append(line + "\n")

    lines.extend([
        f"\nRaw data: {raw_file}\n",
        f"Results: {results_file}\n",
    ])
    summary_file.write_text("".join(lines))
    print("Saved raw data:", raw_file)
    print("Saved results:", results_file)
    print("Saved summary:", summary_file)
    print("Training data and model were NOT modified.")


if __name__ == "__main__":
    main()
