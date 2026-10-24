#!/usr/bin/env python3
"""
CONTROLLED MPU6050 ORIENTATION / AXIS STUDY COLLECTOR

Purpose:
    Collect a balanced 0-9 dataset for one controlled orientation or
    sensor-mount condition at a time.

Default target:
    10 digits
    5 recordings per digit
    50 recordings per orientation session

Important experimental rule:
    Change only ONE factor per session.

    Usage-orientation study:
        Keep the sensor mounted the same; change grip / roll / writing plane.

    Sensor-mount study:
        Keep grip and writing plane consistent; change physical sensor-axis
        alignment relative to the pen.

Each recording:
    100 samples
    6 MPU6050 channels
    50 ms/sample
    approximately 5 seconds

Arduino protocol:
    Python sends uppercase R
    Arduino sends START, 100 rows of ax,ay,az,gx,gy,gz, then END

Output:
    data/numbers/orientation_study/
"""

import glob
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "numbers" / "orientation_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ACQUISITION SETTINGS
# ============================================================

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6
SAMPLE_INTERVAL_MS = 50

TRIALS_PER_DIGIT = 5
DIGITS = [str(i) for i in range(10)]

COUNTDOWN_SECONDS = 3
START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12


# ============================================================
# IDENTIFIER HELPERS
# ============================================================

def clean_identifier(value):
    value = value.strip()

    if not value:
        raise ValueError("Identifier cannot be empty.")

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )


def read_optional_float(prompt):
    value = input(prompt).strip()

    if not value:
        return np.nan

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Expected a number for: {prompt}"
        ) from exc


def normalize_label(value):
    return str(value).replace(".0", "").strip()


# ============================================================
# STUDY METADATA
# ============================================================

def get_study_metadata():
    print()
    print("=" * 72)
    print("ORIENTATION STUDY METADATA")
    print("=" * 72)

    participant_id = clean_identifier(
        input("Participant ID: ")
    )

    default_session = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    session_input = input(
        f"Session ID [ENTER for {default_session}]: "
    ).strip()

    session_id = (
        clean_identifier(session_input)
        if session_input
        else default_session
    )

    dominant_hand = input(
        "Dominant hand (left/right/ambidextrous): "
    ).strip().lower() or "unspecified"

    print()
    print("Study type:")
    print("  1. usage_orientation")
    print("     Keep sensor mounting fixed; change grip/roll/writing plane.")
    print("  2. sensor_mount")
    print("     Keep grip/plane fixed; change physical sensor-axis alignment.")

    while True:
        choice = input("Choose study type [1/2]: ").strip()

        if choice == "1":
            study_type = "usage_orientation"
            break

        if choice == "2":
            study_type = "sensor_mount"
            break

        print("Enter 1 or 2.")

    print()
    print("Suggested orientation IDs:")
    print("  O0_NEUTRAL_FRONT")
    print("  O1_ROLL_CW_90")
    print("  O2_ROLL_CCW_90")
    print("  O3_HORIZONTAL")
    print("  O4_DIAGONAL_45")

    orientation_id = clean_identifier(
        input("Orientation ID: ")
    )

    writing_plane = clean_identifier(
        input(
            "Writing plane "
            "(front_vertical/horizontal/diagonal/other): "
        )
    )

    print()
    print("Suggested sensor mount IDs:")
    print("  MOUNT_X_LONG")
    print("  MOUNT_Y_LONG")
    print("  MOUNT_Z_LONG")
    print("  MOUNT_CURRENT")

    sensor_mount_id = clean_identifier(
        input("Sensor mount ID: ")
    )

    pen_axis_alignment = clean_identifier(
        input(
            "MPU6050 axis aligned with pen length "
            "(X/Y/Z/current): "
        )
    ).upper()

    roll_deg = read_optional_float(
        "Approximate roll angle in degrees [optional]: "
    )

    pitch_deg = read_optional_float(
        "Approximate pitch angle in degrees [optional]: "
    )

    yaw_deg = read_optional_float(
        "Approximate yaw angle in degrees [optional]: "
    )

    notes = input(
        "Short condition notes [optional]: "
    ).strip()

    output_file = (
        OUTPUT_DIR
        / (
            f"numbers_0_9_orientation_"
            f"{participant_id}_"
            f"{session_id}_"
            f"{orientation_id}.csv"
        )
    )

    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "dominant_hand": dominant_hand,
        "study_type": study_type,
        "orientation_id": orientation_id,
        "writing_plane": writing_plane,
        "sensor_mount_id": sensor_mount_id,
        "pen_axis_alignment": pen_axis_alignment,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "condition_notes": notes,
        "output_file": output_file,
    }


# ============================================================
# SERIAL PORT
# ============================================================

def find_serial_ports():
    ports = []

    for pattern in [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]:
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

        raise RuntimeError(
            "Reconnect the Arduino and rerun the script."
        )

    print()
    print("=" * 72)
    print("SERIAL PORT")
    print("=" * 72)

    if len(ports) == 1:
        print("Using:", ports[0])
        return ports[0]

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
    ser = serial.Serial(
        port,
        BAUD_RATE,
        timeout=1,
    )

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


# ============================================================
# COUNTDOWN / CAPTURE
# ============================================================

def countdown():
    print()
    print("Get ready...")

    for number in range(
        COUNTDOWN_SECONDS,
        0,
        -1,
    ):
        print(number)
        time.sleep(1)

    print()
    print("GO — Recording now!")


def wait_for_start(ser):
    deadline = time.monotonic() + START_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        if line == "START":
            return True

    return False


def capture_recording(ser):
    ser.reset_input_buffer()

    ser.write(b"R")
    ser.flush()

    if not wait_for_start(ser):
        raise TimeoutError(
            "Arduino did not send START."
        )

    samples = []
    start_time = time.monotonic()
    saw_end = False

    while (
        time.monotonic() - start_time
        < RECORDING_TIMEOUT_SECONDS
    ):
        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        if not line:
            continue

        if line == "END":
            saw_end = True
            break

        parts = line.split(",")

        if len(parts) != NUM_CHANNELS:
            continue

        try:
            row = [
                float(value)
                for value in parts
            ]
        except ValueError:
            continue

        if not np.all(np.isfinite(row)):
            continue

        samples.append(row)

    elapsed = time.monotonic() - start_time

    if not saw_end:
        raise TimeoutError(
            "Timed out before receiving END."
        )

    recording = np.asarray(
        samples,
        dtype=float,
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS,
    ):
        raise RuntimeError(
            f"Expected ({NUM_SAMPLES}, {NUM_CHANNELS}), "
            f"got {recording.shape}."
        )

    return recording, elapsed


# ============================================================
# RECORDING / CSV HELPERS
# ============================================================

def sensor_summary(recording):
    acceleration = recording[:, 0:3]

    magnitude = np.sqrt(
        np.sum(
            acceleration ** 2,
            axis=1,
        )
    )

    return {
        "mean": float(np.mean(magnitude)),
        "median": float(np.median(magnitude)),
        "min": float(np.min(magnitude)),
        "max": float(np.max(magnitude)),
    }


def make_row(
    metadata,
    digit,
    trial_number,
    recording,
    elapsed,
):
    row = {
        "participant_id": metadata["participant_id"],
        "session_id": metadata["session_id"],
        "dominant_hand": metadata["dominant_hand"],
        "study_type": metadata["study_type"],
        "orientation_id": metadata["orientation_id"],
        "writing_plane": metadata["writing_plane"],
        "sensor_mount_id": metadata["sensor_mount_id"],
        "pen_axis_alignment": metadata["pen_axis_alignment"],
        "roll_deg": metadata["roll_deg"],
        "pitch_deg": metadata["pitch_deg"],
        "yaw_deg": metadata["yaw_deg"],
        "condition_notes": metadata["condition_notes"],
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "actual": digit,
        "trial_number": trial_number,
        "capture_seconds": elapsed,
    }

    for sample_index in range(NUM_SAMPLES):
        for channel in range(NUM_CHANNELS):
            row[
                f"value_{sample_index}_{channel}"
            ] = float(
                recording[
                    sample_index,
                    channel,
                ]
            )

    return row


def load_existing(output_file):
    if not output_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(output_file)

    print()
    print("Existing orientation-session file found.")
    print("Rows:", len(df))
    print("Collection will resume.")

    return df


def completed_count(df, digit):
    if (
        len(df) == 0
        or "actual" not in df.columns
    ):
        return 0

    labels = df["actual"].map(
        normalize_label
    )

    return int((labels == digit).sum())


def save_rows(rows, output_file):
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return df


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_dataset(df, metadata):
    print()
    print("=" * 72)
    print("ORIENTATION SESSION VALIDATION")
    print("=" * 72)

    print("Participant:", metadata["participant_id"])
    print("Session:", metadata["session_id"])
    print("Study type:", metadata["study_type"])
    print("Orientation:", metadata["orientation_id"])
    print("Writing plane:", metadata["writing_plane"])
    print("Sensor mount:", metadata["sensor_mount_id"])
    print("Pen-axis alignment:", metadata["pen_axis_alignment"])
    print("Rows:", len(df))

    labels = df["actual"].map(
        normalize_label
    )

    correct_counts = True

    print()
    print("Class counts:")

    for digit in DIGITS:
        count = int((labels == digit).sum())

        print(
            f"{digit}: "
            f"{count}/"
            f"{TRIALS_PER_DIGIT}"
        )

        if count != TRIALS_PER_DIGIT:
            correct_counts = False

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith("value_")
    ]

    numeric = (
        df[sensor_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    missing = int(
        numeric.isna().sum().sum()
    )

    values = numeric.to_numpy(
        dtype=float
    )

    finite = bool(
        np.all(np.isfinite(values))
    )

    duplicates = int(
        pd.DataFrame(values)
        .duplicated()
        .sum()
    )

    print()
    print("Sensor columns:", len(sensor_columns))
    print("Missing values:", missing)
    print("All finite:", finite)
    print("Exact duplicate recordings:", duplicates)

    if len(df):
        recordings = values.reshape(
            len(df),
            NUM_SAMPLES,
            NUM_CHANNELS,
        )

        acceleration = recordings[:, :, 0:3]

        magnitude = np.sqrt(
            np.sum(
                acceleration ** 2,
                axis=2,
            )
        )

        print()
        print(
            "Acceleration magnitude mean:",
            f"{magnitude.mean():.4f}",
            "m/s^2",
        )

        print(
            "Acceleration magnitude median:",
            f"{np.median(magnitude):.4f}",
            "m/s^2",
        )

    expected_total = (
        len(DIGITS)
        * TRIALS_PER_DIGIT
    )

    passed = (
        len(df) == expected_total
        and correct_counts
        and len(sensor_columns) == 600
        and missing == 0
        and finite
        and duplicates == 0
    )

    print()

    if passed:
        print("ORIENTATION SESSION: PASS")
    else:
        print(
            "ORIENTATION SESSION: "
            "INCOMPLETE / CHECK REQUIRED"
        )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("MPU6050 0-9 ORIENTATION STUDY")
    print("=" * 72)

    print()
    print(
        f"Target: {len(DIGITS)} digits x "
        f"{TRIALS_PER_DIGIT} trials"
    )

    print(
        f"Total: "
        f"{len(DIGITS) * TRIALS_PER_DIGIT} "
        "recordings per condition"
    )

    print()
    print(
        "Change only one experimental "
        "factor during this session."
    )

    metadata = get_study_metadata()
    output_file = metadata["output_file"]

    print()
    print("Output file:")
    print(output_file)

    existing_df = load_existing(
        output_file
    )

    rows = (
        existing_df.to_dict(
            orient="records"
        )
        if len(existing_df)
        else []
    )

    port = choose_serial_port()

    print()
    print("Opening:", port)

    ser = open_serial(port)

    print("Serial connection ready.")

    try:
        for digit in DIGITS:
            current_df = pd.DataFrame(rows)

            already_done = completed_count(
                current_df,
                digit,
            )

            if already_done >= TRIALS_PER_DIGIT:
                print()
                print(
                    f"Digit {digit}: "
                    "already complete."
                )
                continue

            print()
            print("=" * 72)
            print(f"DIGIT {digit}")
            print("=" * 72)
            print(
                f"Completed: {already_done}/"
                f"{TRIALS_PER_DIGIT}"
            )

            for trial in range(
                already_done + 1,
                TRIALS_PER_DIGIT + 1,
            ):
                while True:
                    print()
                    print(
                        f"Digit {digit} — "
                        f"Trial {trial}/"
                        f"{TRIALS_PER_DIGIT}"
                    )

                    print(
                        "Maintain the declared "
                        "orientation condition."
                    )

                    answer = input(
                        "Press ENTER when ready "
                        "or q to quit: "
                    ).strip().lower()

                    if answer == "q":
                        raise KeyboardInterrupt

                    countdown()

                    try:
                        recording, elapsed = (
                            capture_recording(ser)
                        )
                    except Exception as exc:
                        print()
                        print("CAPTURE FAILED:")
                        print(exc)
                        print()
                        print("Trial was NOT saved.")

                        action = input(
                            "Press ENTER to retry, "
                            "r to reconnect serial, "
                            "or q to quit: "
                        ).strip().lower()

                        if action == "q":
                            raise KeyboardInterrupt

                        if action == "r":
                            ser = reconnect_serial(
                                ser
                            )

                        continue

                    summary = sensor_summary(
                        recording
                    )

                    print()
                    print("Samples:", recording.shape[0])
                    print(
                        "Capture duration:",
                        f"{elapsed:.3f} seconds",
                    )
                    print(
                        "Acceleration magnitude mean:",
                        f"{summary['mean']:.4f}",
                    )
                    print(
                        "Acceleration magnitude median:",
                        f"{summary['median']:.4f}",
                    )

                    if not (
                        5.0
                        <= summary["median"]
                        <= 15.0
                    ):
                        print()
                        print(
                            "WARNING: acceleration "
                            "magnitude looks unusual."
                        )

                        keep = input(
                            "Save anyway? (y/n): "
                        ).strip().lower()

                        if keep != "y":
                            print("Trial discarded.")
                            continue

                    rows.append(
                        make_row(
                            metadata,
                            digit,
                            trial,
                            recording,
                            elapsed,
                        )
                    )

                    df = save_rows(
                        rows,
                        output_file,
                    )

                    print()
                    print("SAVED.")
                    print(
                        "Current orientation rows:",
                        len(df),
                    )

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
        final_df = save_rows(
            rows,
            output_file,
        )

        validate_dataset(
            final_df,
            metadata,
        )

        print()
        print("Saved orientation dataset:")
        print(output_file)


if __name__ == "__main__":
    main()
