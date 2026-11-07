#!/usr/bin/env python3
"""
CONTROLLED A-Z MPU6050 BOARD-POSE COLLECTOR

Collect one board pose per session:
    BOARD_FACE_UP
    BOARD_FACE_DOWN
    BOARD_FACE_VERTICAL

Default target:
    26 letters x 5 trials = 130 recordings per pose/session

Each recording:
    100 samples x 6 channels
    50 ms/sample
    approximately 5 seconds

Arduino protocol:
    Python sends uppercase R
    Arduino sends START, 100 CSV rows, then END

Output:
    data/letters/board_pose_study/

The script resumes interrupted sessions and never modifies historical
A-L or M-Z datasets.
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
OUTPUT_DIR = REPO_ROOT / "data" / "letters" / "board_pose_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BAUD_RATE = 115200
NUM_SAMPLES = 100
NUM_CHANNELS = 6
SAMPLE_INTERVAL_MS = 50
TRIALS_PER_LETTER = 5
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

COUNTDOWN_SECONDS = 3
START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12

POSES = {
    "1": "BOARD_FACE_UP",
    "2": "BOARD_FACE_DOWN",
    "3": "BOARD_FACE_VERTICAL",
}


def clean_identifier(value):
    value = value.strip()
    if not value:
        raise ValueError("Identifier cannot be empty.")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def normalize_label(value):
    return str(value).strip().upper()


def choose_pose():
    print()
    print("Board pose:")
    print("  1. BOARD_FACE_UP       - board horizontal, right-side up")
    print("  2. BOARD_FACE_DOWN     - board horizontal, upside down")
    print("  3. BOARD_FACE_VERTICAL - board mounted vertically/on edge")

    while True:
        choice = input("Choose board pose [1/2/3]: ").strip()
        if choice in POSES:
            return POSES[choice]
        print("Enter 1, 2, or 3.")


def get_metadata():
    print()
    print("=" * 72)
    print("A-Z BOARD-POSE SESSION METADATA")
    print("=" * 72)

    participant_id = clean_identifier(input("Participant ID: "))

    default_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_entry = input(
        f"Session ID [ENTER for {default_session}]: "
    ).strip()
    session_id = clean_identifier(session_entry) if session_entry else default_session

    dominant_hand = (
        input("Dominant hand (left/right/ambidextrous): ")
        .strip()
        .lower()
        or "unspecified"
    )

    board_pose = choose_pose()

    writing_plane = clean_identifier(
        input(
            "Writing plane (front_vertical/horizontal/diagonal/other): "
        )
    )

    condition_notes = input("Short condition notes [optional]: ").strip()

    output_file = (
        OUTPUT_DIR
        / f"letters_A_Z_{participant_id}_{session_id}_{board_pose}.csv"
    )

    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "dominant_hand": dominant_hand,
        "study_type": "board_mount_pose",
        "orientation_id": board_pose,
        "board_pose": board_pose,
        "writing_plane": writing_plane,
        "sensor_mount_id": f"MOUNT_{board_pose.replace('BOARD_', '')}",
        "pen_axis_alignment": "UNKNOWN",
        "condition_notes": condition_notes,
        "output_file": output_file,
    }


def find_serial_ports():
    ports = []
    for pattern in ["/dev/cu.usbmodem*", "/dev/cu.usbserial*"]:
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def choose_serial_port():
    ports = find_serial_ports()

    if not ports:
        print()
        print("No Arduino serial port found.")
        print("Detected /dev/cu.* devices:")
        for port in sorted(glob.glob("/dev/cu.*")):
            print("  ", port)
        raise RuntimeError("Reconnect the Arduino and rerun the script.")

    if len(ports) == 1:
        print("Using serial port:", ports[0])
        return ports[0]

    print()
    for index, port in enumerate(ports, start=1):
        print(f"{index}. {port}")

    while True:
        answer = input("Choose port number: ").strip()
        try:
            selected = int(answer) - 1
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

    print()
    print("Attempting serial reconnection...")
    time.sleep(2)

    port = choose_serial_port()
    ser = open_serial(port)
    print("Serial reconnected:", port)
    return ser


def countdown():
    print()
    print("Get ready...")
    for number in range(COUNTDOWN_SECONDS, 0, -1):
        print(number)
        time.sleep(1)
    print()
    print("GO - Recording now!")


def wait_for_start(ser):
    deadline = time.monotonic() + START_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if line == "START":
            return True

    return False


def capture_recording(ser):
    ser.reset_input_buffer()
    ser.write(b"R")
    ser.flush()

    if not wait_for_start(ser):
        raise TimeoutError("Arduino did not send START.")

    samples = []
    start_time = time.monotonic()
    saw_end = False

    while time.monotonic() - start_time < RECORDING_TIMEOUT_SECONDS:
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

    elapsed = time.monotonic() - start_time

    if not saw_end:
        raise TimeoutError("Timed out before receiving END.")

    recording = np.asarray(samples, dtype=float)

    if recording.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise RuntimeError(
            f"Expected ({NUM_SAMPLES}, {NUM_CHANNELS}), got {recording.shape}."
        )

    return recording, elapsed


def sensor_summary(recording):
    magnitude = np.linalg.norm(recording[:, 0:3], axis=1)
    return {
        "mean": float(np.mean(magnitude)),
        "median": float(np.median(magnitude)),
        "min": float(np.min(magnitude)),
        "max": float(np.max(magnitude)),
    }


def make_row(metadata, letter, trial_number, recording, elapsed):
    row = {
        "participant_id": metadata["participant_id"],
        "session_id": metadata["session_id"],
        "dominant_hand": metadata["dominant_hand"],
        "study_type": metadata["study_type"],
        "orientation_id": metadata["orientation_id"],
        "board_pose": metadata["board_pose"],
        "writing_plane": metadata["writing_plane"],
        "sensor_mount_id": metadata["sensor_mount_id"],
        "pen_axis_alignment": metadata["pen_axis_alignment"],
        "condition_notes": metadata["condition_notes"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actual": letter,
        "trial_number": trial_number,
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
    print()
    print("Existing A-Z pose-session file found.")
    print("Rows:", len(df))
    print("Collection will resume.")

    checks = {
        "participant_id": metadata["participant_id"],
        "session_id": metadata["session_id"],
        "orientation_id": metadata["orientation_id"],
    }

    for column, expected in checks.items():
        if column not in df.columns:
            raise RuntimeError(f"Existing file is missing metadata column: {column}")
        actual_values = set(df[column].astype(str).unique())
        if actual_values != {str(expected)}:
            raise RuntimeError(
                f"Existing file metadata mismatch for {column}: {actual_values}"
            )

    return df


def completed_count(df, letter):
    if len(df) == 0 or "actual" not in df.columns:
        return 0
    labels = df["actual"].map(normalize_label)
    return int((labels == letter).sum())


def save_rows(rows, output_file):
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return df


def validate_dataset(df, metadata):
    print()
    print("=" * 72)
    print("A-Z BOARD-POSE SESSION VALIDATION")
    print("=" * 72)
    print("Participant:", metadata["participant_id"])
    print("Session:", metadata["session_id"])
    print("Board pose:", metadata["board_pose"])
    print("Writing plane:", metadata["writing_plane"])
    print("Rows:", len(df))

    labels = df["actual"].map(normalize_label)
    correct_counts = True

    print()
    print("Class counts:")
    for letter in LETTERS:
        count = int((labels == letter).sum())
        print(f"{letter}: {count}/{TRIALS_PER_LETTER}")
        if count != TRIALS_PER_LETTER:
            correct_counts = False

    sensor_columns = [
        column for column in df.columns if column.startswith("value_")
    ]
    numeric = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().sum().sum())
    values = numeric.to_numpy(dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    duplicates = int(pd.DataFrame(values).duplicated().sum())

    print()
    print("Sensor columns:", len(sensor_columns))
    print("Missing values:", missing)
    print("All finite:", finite)
    print("Exact duplicate recordings:", duplicates)

    if len(df) and len(sensor_columns) == NUM_SAMPLES * NUM_CHANNELS:
        recordings = values.reshape(len(df), NUM_SAMPLES, NUM_CHANNELS)
        magnitude = np.linalg.norm(recordings[:, :, 0:3], axis=2)
        print()
        print(f"Acceleration magnitude mean: {magnitude.mean():.4f} m/s^2")
        print(
            f"Acceleration magnitude median: {np.median(magnitude):.4f} m/s^2"
        )

    expected_total = len(LETTERS) * TRIALS_PER_LETTER
    passed = (
        len(df) == expected_total
        and correct_counts
        and len(sensor_columns) == NUM_SAMPLES * NUM_CHANNELS
        and missing == 0
        and finite
        and duplicates == 0
    )

    print()
    print("A-Z BOARD-POSE SESSION:", "PASS" if passed else "CHECK REQUIRED")
    return passed


def main():
    print()
    print("=" * 72)
    print("A-Z MPU6050 BOARD-POSE DATA COLLECTION")
    print("=" * 72)
    print(f"Target: {len(LETTERS)} letters x {TRIALS_PER_LETTER} trials")
    print(f"Total: {len(LETTERS) * TRIALS_PER_LETTER} recordings")
    print(f"Format: {NUM_SAMPLES} samples x {NUM_CHANNELS} channels")

    metadata = get_metadata()
    output_file = metadata["output_file"]

    print()
    print("Output file:")
    print(output_file)

    existing_df = load_existing(output_file, metadata)
    rows = existing_df.to_dict(orient="records") if len(existing_df) else []

    port = choose_serial_port()
    ser = open_serial(port)
    print("Serial connection ready.")

    try:
        for letter in LETTERS:
            already_done = completed_count(pd.DataFrame(rows), letter)

            if already_done >= TRIALS_PER_LETTER:
                print(f"Letter {letter}: already complete.")
                continue

            print()
            print("=" * 72)
            print(f"LETTER {letter}")
            print("=" * 72)

            for trial in range(already_done + 1, TRIALS_PER_LETTER + 1):
                while True:
                    print()
                    print(f"Letter {letter} - Trial {trial}/{TRIALS_PER_LETTER}")
                    answer = input(
                        "Press ENTER when ready or q to quit: "
                    ).strip().lower()

                    if answer == "q":
                        raise KeyboardInterrupt

                    countdown()

                    try:
                        recording, elapsed = capture_recording(ser)
                    except Exception as exc:
                        print()
                        print("CAPTURE FAILED:")
                        print(exc)
                        print("Trial was NOT saved.")
                        action = input(
                            "Press ENTER to retry, r to reconnect, or q to quit: "
                        ).strip().lower()
                        if action == "q":
                            raise KeyboardInterrupt
                        if action == "r":
                            ser = reconnect_serial(ser)
                        continue

                    summary = sensor_summary(recording)
                    print()
                    print("Samples:", recording.shape[0])
                    print(f"Capture duration: {elapsed:.3f} seconds")
                    print(f"Acceleration magnitude mean: {summary['mean']:.4f}")
                    print(f"Acceleration magnitude median: {summary['median']:.4f}")

                    if not 5.0 <= summary["median"] <= 15.0:
                        keep = input(
                            "Unusual acceleration magnitude. Save anyway? (y/n): "
                        ).strip().lower()
                        if keep != "y":
                            print("Trial discarded.")
                            continue

                    rows.append(
                        make_row(
                            metadata,
                            letter,
                            trial,
                            recording,
                            elapsed,
                        )
                    )
                    df = save_rows(rows, output_file)
                    print("SAVED. Current rows:", len(df))
                    break

    except KeyboardInterrupt:
        print()
        print("Collection stopped.")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("Serial connection closed.")

    if rows:
        final_df = save_rows(rows, output_file)
        validate_dataset(final_df, metadata)
        print("Saved A-Z board-pose dataset:")
        print(output_file)


if __name__ == "__main__":
    main()
