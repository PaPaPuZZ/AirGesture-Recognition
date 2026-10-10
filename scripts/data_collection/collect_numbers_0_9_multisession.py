#!/usr/bin/env python3
"""
MULTI-SESSION NUMBERS 0-9 DATA COLLECTOR

Purpose:
    Collect a balanced, fresh batch of digit gestures for improved
    cross-session and future cross-participant generalization.

Default collection target:
    10 recordings per digit
    10 digits
    100 recordings total

Each recording:
    100 samples
    6 channels
    50 ms/sample
    ~5 second gesture window

Arduino protocol:
    Python sends uppercase R
    Arduino sends:
        START
        100 rows of ax,ay,az,gx,gy,gz
        END

Metadata saved with every recording:
    participant_id
    session_id
    dominant_hand
    actual
    trial_number
    timestamp

Important:
    - This script creates one CSV per participant/session.
    - It can resume an interrupted session.
    - It does NOT modify the existing 215-recording training dataset.
    - Certification/holdout CSVs should remain test-only and should NOT
      be merged into training.
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
OUTPUT_DIR = REPO_ROOT / "data" / "numbers" / "multisession"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ACQUISITION SETTINGS
# ============================================================

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

SAMPLE_INTERVAL_MS = 50
EXPECTED_WINDOW_SECONDS = (
    NUM_SAMPLES * SAMPLE_INTERVAL_MS / 1000.0
)

TRIALS_PER_DIGIT = 10

DIGITS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12


# ============================================================
# HELPERS
# ============================================================

def clean_identifier(value):
    value = value.strip()

    if not value:
        raise ValueError("Identifier cannot be empty.")

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value
    )


def normalize_label(value):
    return (
        str(value)
        .replace(".0", "")
        .strip()
    )


# ============================================================
# SESSION METADATA
# ============================================================

def get_session_metadata():

    print()
    print("=" * 72)
    print("SESSION METADATA")
    print("=" * 72)

    participant_id = clean_identifier(
        input(
            "Participant ID "
            "(example: nicholas, p01, friend01): "
        )
    )

    default_session = (
        datetime.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    session_entry = input(
        f"Session ID "
        f"[press ENTER for {default_session}]: "
    ).strip()

    if session_entry:

        session_id = clean_identifier(
            session_entry
        )

    else:

        session_id = default_session

    dominant_hand = input(
        "Dominant hand "
        "(left/right/ambidextrous): "
    ).strip().lower()

    if dominant_hand not in {
        "left",
        "right",
        "ambidextrous",
    }:

        dominant_hand = (
            dominant_hand
            if dominant_hand
            else "unspecified"
        )

    output_file = (
        OUTPUT_DIR
        / (
            f"numbers_0_9_"
            f"{participant_id}_"
            f"{session_id}.csv"
        )
    )

    return (
        participant_id,
        session_id,
        dominant_hand,
        output_file,
    )


# ============================================================
# SERIAL
# ============================================================

def find_serial_ports():

    ports = []

    for pattern in [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]:

        ports.extend(
            glob.glob(pattern)
        )

    return sorted(
        set(ports)
    )


def choose_serial_port():

    ports = find_serial_ports()

    if not ports:

        print()
        print("No Arduino serial port found.")
        print("Detected /dev/cu.* devices:")

        all_ports = sorted(
            glob.glob(
                "/dev/cu.*"
            )
        )

        for port in all_ports:

            print(
                "  ",
                port
            )

        raise RuntimeError(
            "Reconnect the Arduino and "
            "run the script again."
        )

    print()
    print("=" * 72)
    print("SERIAL PORT")
    print("=" * 72)

    if len(ports) == 1:

        print(
            "Using:",
            ports[0]
        )

        return ports[0]

    for index, port in enumerate(
        ports,
        start=1
    ):

        print(
            f"{index}. {port}"
        )

    while True:

        answer = input(
            "Choose port number: "
        ).strip()

        try:

            selected = (
                int(answer) - 1
            )

            if (
                0
                <= selected
                < len(ports)
            ):

                return ports[
                    selected
                ]

        except ValueError:

            pass

        print(
            "Invalid selection."
        )


def open_serial(port):

    ser = serial.Serial(
        port,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(
        2
    )

    ser.reset_input_buffer()

    return ser


def reconnect_serial(old_ser=None):

    if old_ser is not None:

        try:

            old_ser.close()

        except Exception:

            pass

    print()
    print(
        "Attempting serial reconnection..."
    )

    time.sleep(
        2
    )

    port = choose_serial_port()

    ser = open_serial(
        port
    )

    print(
        "Serial reconnected:",
        port
    )

    return ser


# ============================================================
# COUNTDOWN
# ============================================================

def countdown():

    print()
    print(
        "Get ready..."
    )

    for number in range(
        COUNTDOWN_SECONDS,
        0,
        -1
    ):

        print(
            number
        )

        time.sleep(
            1
        )

    print()
    print(
        "GO — Recording now!"
    )


# ============================================================
# CAPTURE
# ============================================================

def wait_for_start(ser):

    deadline = (
        time.monotonic()
        + START_TIMEOUT_SECONDS
    )

    while (
        time.monotonic()
        < deadline
    ):

        raw = ser.readline()

        if not raw:

            continue

        line = raw.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if line == "START":

            return True

    return False


def capture_recording(ser):

    ser.reset_input_buffer()

    ser.write(
        b"R"
    )

    ser.flush()

    if not wait_for_start(
        ser
    ):

        raise TimeoutError(
            "Arduino did not send START."
        )

    samples = []

    start_time = (
        time.monotonic()
    )

    saw_end = False

    while (
        time.monotonic()
        - start_time
        < RECORDING_TIMEOUT_SECONDS
    ):

        raw = ser.readline()

        if not raw:

            continue

        line = raw.decode(
            "utf-8",
            errors="ignore"
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

        if not np.all(
            np.isfinite(
                row
            )
        ):

            continue

        samples.append(
            row
        )

    elapsed = (
        time.monotonic()
        - start_time
    )

    if not saw_end:

        raise TimeoutError(
            "Timed out before receiving END."
        )

    recording = np.asarray(
        samples,
        dtype=float
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS
    ):

        raise RuntimeError(
            f"Expected "
            f"({NUM_SAMPLES}, {NUM_CHANNELS}), "
            f"got {recording.shape}."
        )

    return (
        recording,
        elapsed,
    )


# ============================================================
# RECORDING QUALITY
# ============================================================

def sensor_summary(recording):

    acceleration = recording[
        :,
        0:3
    ]

    magnitude = np.sqrt(
        np.sum(
            acceleration ** 2,
            axis=1
        )
    )

    return {
        "mean": float(
            np.mean(magnitude)
        ),
        "median": float(
            np.median(magnitude)
        ),
        "min": float(
            np.min(magnitude)
        ),
        "max": float(
            np.max(magnitude)
        ),
    }


# ============================================================
# CSV
# ============================================================

def make_row(
    participant_id,
    session_id,
    dominant_hand,
    digit,
    trial_number,
    recording,
    elapsed,
):

    row = {
        "participant_id":
            participant_id,

        "session_id":
            session_id,

        "dominant_hand":
            dominant_hand,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "actual":
            digit,

        "trial_number":
            trial_number,

        "capture_seconds":
            elapsed,
    }

    for sample_index in range(
        NUM_SAMPLES
    ):

        for channel in range(
            NUM_CHANNELS
        ):

            row[
                f"value_"
                f"{sample_index}_"
                f"{channel}"
            ] = float(
                recording[
                    sample_index,
                    channel
                ]
            )

    return row


def load_existing(output_file):

    if not output_file.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        output_file
    )

    print()
    print(
        "Existing session file found."
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Collection will resume."
    )

    return df


def completed_count(
    df,
    digit
):

    if (
        len(df) == 0
        or "actual"
        not in df.columns
    ):

        return 0

    labels = (
        df["actual"]
        .map(
            normalize_label
        )
    )

    return int(
        (
            labels == digit
        ).sum()
    )


def save_rows(
    rows,
    output_file
):

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        output_file,
        index=False
    )

    return df


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_dataset(
    df,
    participant_id,
    session_id,
):

    print()
    print("=" * 72)
    print("SESSION DATASET VALIDATION")
    print("=" * 72)

    print(
        "Participant:",
        participant_id
    )

    print(
        "Session:",
        session_id
    )

    print(
        "Rows:",
        len(df)
    )

    labels = (
        df["actual"]
        .map(
            normalize_label
        )
    )

    print()
    print(
        "Class counts:"
    )

    correct_counts = True

    for digit in DIGITS:

        count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"{digit}: "
            f"{count}/"
            f"{TRIALS_PER_DIGIT}"
        )

        if (
            count
            != TRIALS_PER_DIGIT
        ):

            correct_counts = False

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(
            "value_"
        )
    ]

    print()
    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    numeric = (
        df[
            sensor_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    missing = int(
        numeric
        .isna()
        .sum()
        .sum()
    )

    values = numeric.to_numpy(
        dtype=float
    )

    finite = bool(
        np.all(
            np.isfinite(
                values
            )
        )
    )

    duplicates = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "All finite:",
        finite
    )

    print(
        "Exact duplicate recordings:",
        duplicates
    )

    if len(df):

        recordings = values.reshape(
            len(df),
            NUM_SAMPLES,
            NUM_CHANNELS
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

        print(
            "SESSION DATASET: PASS"
        )

    else:

        print(
            "SESSION DATASET: "
            "INCOMPLETE / CHECK REQUIRED"
        )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 0-9 "
        "MULTI-SESSION DATA COLLECTION"
    )
    print("=" * 72)

    print()
    print(
        "Target:"
    )

    print(
        f"{TRIALS_PER_DIGIT} recordings "
        "per digit"
    )

    print(
        f"{len(DIGITS) * TRIALS_PER_DIGIT} "
        "recordings total"
    )

    print()
    print(
        "Recording format:"
    )

    print(
        f"{NUM_SAMPLES} samples x "
        f"{NUM_CHANNELS} channels"
    )

    print(
        f"{SAMPLE_INTERVAL_MS} ms/sample"
    )

    print(
        f"~{EXPECTED_WINDOW_SECONDS:.2f} "
        "seconds per gesture"
    )

    (
        participant_id,
        session_id,
        dominant_hand,
        output_file,
    ) = get_session_metadata()

    print()
    print(
        "Output file:"
    )

    print(
        output_file
    )

    existing_df = load_existing(
        output_file
    )

    rows = (
        existing_df
        .to_dict(
            orient="records"
        )
        if len(existing_df)
        else []
    )

    # Validate that a resumed file belongs to this session.
    if len(existing_df):

        if (
            "participant_id"
            in existing_df.columns
        ):

            participants = set(
                existing_df[
                    "participant_id"
                ].astype(str)
            )

            if participants != {
                participant_id
            }:

                raise RuntimeError(
                    "Existing file participant "
                    "does not match."
                )

    port = choose_serial_port()

    print()
    print(
        "Opening:",
        port
    )

    ser = open_serial(
        port
    )

    print(
        "Serial connection ready."
    )

    try:

        for digit in DIGITS:

            current_df = pd.DataFrame(
                rows
            )

            already_done = (
                completed_count(
                    current_df,
                    digit
                )
            )

            if (
                already_done
                >= TRIALS_PER_DIGIT
            ):

                print()
                print(
                    f"Digit {digit}: "
                    "already complete."
                )

                continue

            print()
            print("=" * 72)
            print(
                f"DIGIT {digit}"
            )
            print("=" * 72)

            print(
                f"Completed: "
                f"{already_done}/"
                f"{TRIALS_PER_DIGIT}"
            )

            for trial in range(
                already_done + 1,
                TRIALS_PER_DIGIT + 1
            ):

                while True:

                    print()
                    print(
                        f"Digit {digit} — "
                        f"Trial "
                        f"{trial}/"
                        f"{TRIALS_PER_DIGIT}"
                    )

                    print(
                        "Use a natural variation "
                        "of the gesture."
                    )

                    answer = input(
                        "Press ENTER when ready "
                        "or q to quit: "
                    ).strip().lower()

                    if answer == "q":

                        raise KeyboardInterrupt

                    countdown()

                    try:

                        (
                            recording,
                            elapsed,
                        ) = capture_recording(
                            ser
                        )

                    except Exception as exc:

                        print()
                        print(
                            "CAPTURE FAILED:"
                        )

                        print(
                            exc
                        )

                        print()
                        print(
                            "Trial was NOT saved."
                        )

                        reconnect = input(
                            "Press ENTER to retry, "
                            "r to reconnect serial, "
                            "or q to quit: "
                        ).strip().lower()

                        if reconnect == "q":

                            raise KeyboardInterrupt

                        if reconnect == "r":

                            ser = (
                                reconnect_serial(
                                    ser
                                )
                            )

                        continue

                    summary = sensor_summary(
                        recording
                    )

                    print()
                    print(
                        "Samples:",
                        recording.shape[0]
                    )

                    print(
                        "Capture duration:",
                        f"{elapsed:.3f} seconds"
                    )

                    print(
                        "Acceleration magnitude:"
                    )

                    print(
                        "  mean:",
                        f"{summary['mean']:.4f}"
                    )

                    print(
                        "  median:",
                        f"{summary['median']:.4f}"
                    )

                    print(
                        "  min:",
                        f"{summary['min']:.4f}"
                    )

                    print(
                        "  max:",
                        f"{summary['max']:.4f}"
                    )

                    if not (
                        5.0
                        <= summary[
                            "median"
                        ]
                        <= 15.0
                    ):

                        print()
                        print(
                            "WARNING: acceleration "
                            "magnitude looks unusual."
                        )

                        keep = input(
                            "Save anyway? "
                            "(y/n): "
                        ).strip().lower()

                        if keep != "y":

                            print(
                                "Trial discarded."
                            )

                            continue

                    rows.append(
                        make_row(
                            participant_id,
                            session_id,
                            dominant_hand,
                            digit,
                            trial,
                            recording,
                            elapsed,
                        )
                    )

                    df = save_rows(
                        rows,
                        output_file
                    )

                    print()
                    print(
                        "SAVED."
                    )

                    print(
                        "Current session rows:",
                        len(df)
                    )

                    break

    except KeyboardInterrupt:

        print()
        print(
            "Collection stopped."
        )

    finally:

        try:

            ser.close()

        except Exception:

            pass

        print(
            "Serial connection closed."
        )

    if rows:

        final_df = save_rows(
            rows,
            output_file
        )

        validate_dataset(
            final_df,
            participant_id,
            session_id,
        )

        print()
        print(
            "Saved session dataset:"
        )

        print(
            output_file
        )


if __name__ == "__main__":
    main()
