#!/usr/bin/env python3
"""
CONTROLLED 40 MS / 4 S BOARD-POSE COLLECTOR

Collect either:
  - numbers 0-9, or
  - letters A-Z

One board pose is collected per session:
  BOARD_FACE_UP
  BOARD_FACE_DOWN
  BOARD_FACE_VERTICAL

Protocol:
  100 samples x 6 MPU6050 channels
  40 ms/sample = 25 Hz
  approximately 4.00 seconds

Required Arduino firmware:
  arduino/mpu6050_capture_40ms_4s.ino

The 40 ms datasets are stored separately from the historical 50 ms datasets.
Interrupted sessions resume safely and failed trials are never saved.
"""

import glob
import re
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
SAMPLE_INTERVAL_MS = 40
WINDOW_SECONDS = 4.0
TRIALS_PER_CLASS = 5
COUNTDOWN_SECONDS = 3
START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 10
PROTOCOL_ID = "MPU6050_100X6_40MS_4S"

POSES = {
    "1": "BOARD_FACE_UP",
    "2": "BOARD_FACE_DOWN",
    "3": "BOARD_FACE_VERTICAL",
}

TASKS = {
    "1": {
        "task_id": "numbers_0_9",
        "classes": list("0123456789"),
        "label_name": "Digit",
        "output_dir": REPO_ROOT / "data/numbers/board_pose_study_40ms",
        "prefix": "numbers_0_9_40ms",
    },
    "2": {
        "task_id": "letters_A_Z",
        "classes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "label_name": "Letter",
        "output_dir": REPO_ROOT / "data/letters/board_pose_study_40ms",
        "prefix": "letters_A_Z_40ms",
    },
}


def clean_identifier(value):
    value = value.strip()
    if not value:
        raise ValueError("Identifier cannot be empty.")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def normalize_label(value):
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".0") else text


def choose_from_menu(title, options):
    print()
    print(title)
    for key, value in options.items():
        if isinstance(value, dict):
            print(f"  {key}. {value['task_id']}")
        else:
            print(f"  {key}. {value}")
    while True:
        choice = input("Choose: ").strip()
        if choice in options:
            return choice
        print("Invalid selection.")


def get_metadata():
    print("=" * 72)
    print("40 MS / 4 S BOARD-POSE COLLECTION")
    print("=" * 72)

    task_key = choose_from_menu("Character set:", TASKS)
    task = TASKS[task_key]
    pose_key = choose_from_menu("Board pose:", POSES)
    pose = POSES[pose_key]

    participant_id = clean_identifier(input("Participant ID: "))
    default_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_entry = input(
        f"Session ID [ENTER for {default_session}]: "
    ).strip()
    session_id = clean_identifier(session_entry) if session_entry else default_session
    dominant_hand = (
        input("Dominant hand (left/right/ambidextrous): ").strip().lower()
        or "unspecified"
    )
    writing_plane = clean_identifier(
        input("Writing plane (front_vertical/horizontal/diagonal/other): ")
    )
    notes = input("Short condition notes [optional]: ").strip()

    output_dir = task["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"{task['prefix']}_{participant_id}_{session_id}_{pose}.csv"
    )

    return {
        **task,
        "participant_id": participant_id,
        "session_id": session_id,
        "dominant_hand": dominant_hand,
        "orientation_id": pose,
        "board_pose": pose,
        "writing_plane": writing_plane,
        "condition_notes": notes,
        "output_file": output_file,
    }


def find_serial_ports():
    ports = []
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*"):
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def choose_serial_port():
    ports = find_serial_ports()
    if not ports:
        print("No Arduino serial port found. Detected /dev/cu.* devices:")
        for port in sorted(glob.glob("/dev/cu.*")):
            print("  ", port)
        raise RuntimeError("Reconnect the Arduino and rerun the script.")
    if len(ports) == 1:
        print("Using serial port:", ports[0])
        return ports[0]
    for index, port in enumerate(ports, start=1):
        print(f"{index}. {port}")
    while True:
        try:
            selected = int(input("Choose port number: ").strip()) - 1
            if 0 <= selected < len(ports):
                return ports[selected]
        except ValueError:
            pass
        print("Invalid selection.")


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
    print("Attempting serial reconnection...")
    time.sleep(2)
    port = choose_serial_port()
    ser = open_serial(port)
    print("Serial reconnected:", port)
    return ser


def countdown():
    print("Get ready...")
    for number in range(COUNTDOWN_SECONDS, 0, -1):
        print(number)
        time.sleep(1)
    print("GO - Recording now!")


def wait_for_start(ser):
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        if raw.decode("utf-8", errors="ignore").strip() == "START":
            return True
    return False


def capture_recording(ser):
    ser.reset_input_buffer()
    ser.write(b"R")
    ser.flush()
    if not wait_for_start(ser):
        raise TimeoutError("Arduino did not send START.")

    samples = []
    started = time.monotonic()
    saw_end = False

    while time.monotonic() - started < RECORDING_TIMEOUT_SECONDS:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
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
        raise TimeoutError("Timed out before receiving END.")

    recording = np.asarray(samples, dtype=float)
    if recording.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise RuntimeError(
            f"Expected ({NUM_SAMPLES}, {NUM_CHANNELS}), got {recording.shape}."
        )
    return recording, elapsed


def make_row(metadata, label, trial, recording, elapsed):
    row = {
        "participant_id": metadata["participant_id"],
        "session_id": metadata["session_id"],
        "dominant_hand": metadata["dominant_hand"],
        "task_id": metadata["task_id"],
        "study_type": "board_pose_protocol_comparison",
        "orientation_id": metadata["orientation_id"],
        "board_pose": metadata["board_pose"],
        "writing_plane": metadata["writing_plane"],
        "condition_notes": metadata["condition_notes"],
        "protocol_id": PROTOCOL_ID,
        "num_samples": NUM_SAMPLES,
        "num_channels": NUM_CHANNELS,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "sampling_rate_hz": 1000.0 / SAMPLE_INTERVAL_MS,
        "window_seconds": WINDOW_SECONDS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actual": label,
        "trial_number": trial,
        "capture_seconds": elapsed,
    }
    for sample_index in range(NUM_SAMPLES):
        for channel in range(NUM_CHANNELS):
            row[f"value_{sample_index}_{channel}"] = float(
                recording[sample_index, channel]
            )
    return row


def load_existing(path, metadata):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    print("Existing 40 ms session found. Rows:", len(df))
    for column in ("participant_id", "session_id", "orientation_id", "protocol_id"):
        expected = metadata.get(column, PROTOCOL_ID if column == "protocol_id" else None)
        if column not in df.columns:
            raise RuntimeError(f"Existing file lacks {column}.")
        values = set(df[column].astype(str).unique())
        if values != {str(expected)}:
            raise RuntimeError(f"Metadata mismatch for {column}: {values}")
    return df


def completed_count(df, label):
    if len(df) == 0:
        return 0
    labels = df["actual"].map(normalize_label)
    return int((labels == label).sum())


def save_rows(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def validate_dataset(df, metadata):
    labels = df["actual"].map(normalize_label)
    classes = metadata["classes"]
    expected_total = len(classes) * TRIALS_PER_CLASS
    sensor_columns = [c for c in df.columns if c.startswith("value_")]
    numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)

    print("=" * 72)
    print("40 MS / 4 S SESSION VALIDATION")
    print("=" * 72)
    print("Task:", metadata["task_id"])
    print("Session:", metadata["session_id"])
    print("Board pose:", metadata["board_pose"])
    print("Rows:", len(df))
    print("Class counts:")

    correct_counts = True
    for label in classes:
        count = int((labels == label).sum())
        print(f"{label}: {count}/{TRIALS_PER_CLASS}")
        correct_counts &= count == TRIALS_PER_CLASS

    missing = int(numeric.isna().sum().sum())
    finite = bool(np.all(np.isfinite(values)))
    duplicates = int(pd.DataFrame(values).duplicated().sum())
    print("Sensor columns:", len(sensor_columns))
    print("Missing values:", missing)
    print("All finite:", finite)
    print("Exact duplicate recordings:", duplicates)

    if len(df) and len(sensor_columns) == NUM_SAMPLES * NUM_CHANNELS:
        recordings = values.reshape(len(df), NUM_SAMPLES, NUM_CHANNELS)
        magnitude = np.linalg.norm(recordings[:, :, :3], axis=2)
        print(f"Acceleration magnitude mean: {magnitude.mean():.4f} m/s^2")
        print(f"Acceleration magnitude median: {np.median(magnitude):.4f} m/s^2")
        capture = pd.to_numeric(df["capture_seconds"], errors="coerce")
        print(f"Capture duration mean: {capture.mean():.3f} s")

    protocol_ok = (
        set(pd.to_numeric(df["sample_interval_ms"], errors="coerce").dropna())
        == {SAMPLE_INTERVAL_MS}
        and set(pd.to_numeric(df["window_seconds"], errors="coerce").dropna())
        == {WINDOW_SECONDS}
    )

    passed = (
        len(df) == expected_total
        and correct_counts
        and len(sensor_columns) == NUM_SAMPLES * NUM_CHANNELS
        and missing == 0
        and finite
        and duplicates == 0
        and protocol_ok
    )
    print("40 MS / 4 S SESSION:", "PASS" if passed else "CHECK REQUIRED")
    return passed


def main():
    metadata = get_metadata()
    classes = metadata["classes"]
    print("Output file:", metadata["output_file"])
    print(
        f"Target: {len(classes)} classes x {TRIALS_PER_CLASS} trials = "
        f"{len(classes) * TRIALS_PER_CLASS} recordings"
    )
    print("Protocol: 100 samples, 40 ms/sample, 4.00 seconds")

    existing = load_existing(metadata["output_file"], metadata)
    rows = existing.to_dict(orient="records") if len(existing) else []
    ser = open_serial(choose_serial_port())
    print("Serial connection ready.")

    try:
        for label in classes:
            already_done = completed_count(pd.DataFrame(rows), label)
            if already_done >= TRIALS_PER_CLASS:
                print(f"{metadata['label_name']} {label}: already complete.")
                continue

            for trial in range(already_done + 1, TRIALS_PER_CLASS + 1):
                while True:
                    print(
                        f"{metadata['label_name']} {label} - "
                        f"Trial {trial}/{TRIALS_PER_CLASS}"
                    )
                    answer = input("Press ENTER when ready or q to quit: ").strip().lower()
                    if answer == "q":
                        raise KeyboardInterrupt
                    countdown()
                    try:
                        recording, elapsed = capture_recording(ser)
                    except Exception as exc:
                        print("CAPTURE FAILED:", exc)
                        print("Trial was NOT saved.")
                        action = input(
                            "Press ENTER to retry, r to reconnect, or q to quit: "
                        ).strip().lower()
                        if action == "q":
                            raise KeyboardInterrupt
                        if action == "r":
                            ser = reconnect_serial(ser)
                        continue

                    magnitude = np.linalg.norm(recording[:, :3], axis=1)
                    print("Samples:", len(recording))
                    print(f"Capture duration: {elapsed:.3f} seconds")
                    print(f"Acceleration magnitude median: {np.median(magnitude):.4f}")

                    if not 3.6 <= elapsed <= 4.6:
                        keep = input(
                            "Capture duration is outside expected range. Save? (y/n): "
                        ).strip().lower()
                        if keep != "y":
                            continue

                    rows.append(make_row(metadata, label, trial, recording, elapsed))
                    df = save_rows(rows, metadata["output_file"])
                    print("SAVED. Current rows:", len(df))
                    break
    except KeyboardInterrupt:
        print("Collection stopped.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("Serial connection closed.")

    if rows:
        final_df = save_rows(rows, metadata["output_file"])
        validate_dataset(final_df, metadata)
        print("Saved dataset:", metadata["output_file"])


if __name__ == "__main__":
    main()
