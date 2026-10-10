#!/usr/bin/env python3

"""
MPU6050 NUMBER 1-5 LIVE DATA COLLECTION

Purpose:
    Collect a NEW number-only dataset for:
        1, 2, 3, 4, 5

    10 recordings per number
    100 samples per recording
    6 MPU6050 channels per sample
    600 sensor values per recording

IMPORTANT:
    - Does NOT modify any A-L files.
    - Does NOT retrain any model.
    - Does NOT overwrite the frozen D-L model.
    - Sends the Arduino "R" trigger automatically.
    - Supports BOTH:
          1. 100 serial lines x 6 values
          2. 1 serial line x 600 values
"""

import os
import sys
import time
import glob
import csv
import re

import numpy as np
import pandas as pd

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed.")
    print("Install it with:")
    print("    pip install pyserial")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6
EXPECTED_VALUES = NUM_SAMPLES * NUM_CHANNELS

NUMBERS = ["1", "2", "3", "4", "5"]
TRIALS_PER_NUMBER = 10

OUTPUT_FILE = "numbers_1_5_50ms_training_raw_data.csv"
PROGRESS_FILE = "numbers_1_5_collection_progress.csv"

SERIAL_TIMEOUT = 0.25
CAPTURE_TIMEOUT = 20.0

CHANNEL_NAMES = [
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def divider():
    print("=" * 78)


def section(title):
    print()
    divider()
    print(title)
    divider()
    print()


def flush_print(text=""):
    print(text, flush=True)


# ============================================================
# SERIAL PORT DETECTION
# ============================================================

def find_serial_ports():
    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/cu.wchusbserial*",
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
    ]

    ports = []

    for pattern in patterns:
        ports.extend(glob.glob(pattern))

    return sorted(set(ports))


def choose_serial_port():
    ports = find_serial_ports()

    if not ports:
        print("ERROR: No likely Arduino serial port was detected.")
        print()
        print("Try:")
        print("    ls /dev/cu.*")
        print()
        print("Then reconnect the Arduino and run this script again.")
        sys.exit(1)

    if len(ports) == 1:
        return ports[0]

    print("Available serial ports:")
    for index, port in enumerate(ports, start=1):
        print(f"{index}. {port}")

    print()

    while True:
        choice = input(
            f"Select port [1-{len(ports)}]: "
        ).strip()

        try:
            choice_number = int(choice)

            if 1 <= choice_number <= len(ports):
                return ports[choice_number - 1]

        except ValueError:
            pass

        print("Invalid selection.")


# ============================================================
# SERIAL PARSING
# ============================================================

def extract_numeric_values(line):
    """
    Attempt to extract comma-separated numeric values from
    one Arduino serial line.

    Accepts:
        a,b,c,d,e,f

    or:
        a,b,c,... 600 total values

    Ignores Arduino status messages such as:
        MPU6050 READY
        DONE
        Send R to record another gesture.
    """

    text = line.strip()

    if not text:
        return []

    upper = text.upper()

    ignore_phrases = [
        "READY",
        "DONE",
        "SEND R",
        "RECORD",
        "GESTURE",
        "MPU6050",
    ]

    if any(phrase in upper for phrase in ignore_phrases):
        return []

    # Remove common serial prefixes if present.
    text = re.sub(
        r"^\s*\[Arduino\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # A sensor line should contain commas.
    if "," not in text:
        return []

    pieces = text.split(",")

    values = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        try:
            values.append(float(piece))
        except ValueError:
            return []

    return values


def drain_serial(ser, duration=0.40):
    """
    Clear stale Arduino messages before starting a new trial.
    """
    end_time = time.time() + duration

    while time.time() < end_time:
        try:
            if ser.in_waiting:
                ser.readline()
            else:
                time.sleep(0.02)
        except Exception:
            break


# ============================================================
# TRIAL CAPTURE
# ============================================================

def capture_recording(ser):
    """
    Automatically sends R and captures exactly 600 MPU6050
    values.

    Supports:
        A) one flattened 600-value line
        B) 100 lines containing 6 values each
    """

    drain_serial(ser)

    # Automatically trigger Arduino collection.
    ser.write(b"R\n")
    ser.flush()

    collected = []

    lines_examined = 0
    start_time = time.time()

    while time.time() - start_time < CAPTURE_TIMEOUT:

        try:
            raw = ser.readline()
        except serial.SerialException as exc:
            print(f"\nSERIAL ERROR: {exc}")
            return None

        if not raw:
            continue

        try:
            line = raw.decode(
                "utf-8",
                errors="ignore",
            ).strip()
        except Exception:
            continue

        if not line:
            continue

        lines_examined += 1

        upper = line.upper()

        # Show important Arduino status messages.
        if (
            "DONE" in upper
            or "READY" in upper
            or "SEND R" in upper
        ):
            print(f"[Arduino] {line}")
            continue

        values = extract_numeric_values(line)

        if not values:
            continue

        # ----------------------------------------------------
        # FORMAT A:
        # Arduino sends all 600 values in ONE line.
        # ----------------------------------------------------
        if len(values) == EXPECTED_VALUES:
            collected = values

            print(
                f"Captured flattened recording: "
                f"{len(collected)}/{EXPECTED_VALUES} values."
            )
            break

        # ----------------------------------------------------
        # FORMAT B:
        # Arduino sends one 6-channel sample per line.
        # ----------------------------------------------------
        if len(values) == NUM_CHANNELS:
            collected.extend(values)

            sample_count = len(collected) // NUM_CHANNELS

            if (
                sample_count == 1
                or sample_count % 10 == 0
                or sample_count == NUM_SAMPLES
            ):
                print(
                    f"Samples captured: "
                    f"{sample_count}/{NUM_SAMPLES}",
                    end="\r",
                    flush=True,
                )

            if len(collected) == EXPECTED_VALUES:
                print()
                break

        # If some unexpected comma-separated numeric format
        # appears, ignore it rather than corrupting the trial.

    print()

    if len(collected) != EXPECTED_VALUES:
        print(
            f"ERROR: Captured {len(collected)}/"
            f"{EXPECTED_VALUES} valid sensor values."
        )
        print(
            f"Serial lines examined: {lines_examined}"
        )
        return None

    data = np.asarray(
        collected,
        dtype=float,
    )

    if not np.isfinite(data).all():
        print(
            "ERROR: Recording contains NaN or infinite values."
        )
        return None

    return data


# ============================================================
# DATASET COLUMN CREATION
# ============================================================

def create_sensor_column_names():
    columns = []

    for sample_index in range(NUM_SAMPLES):
        for channel_index in range(NUM_CHANNELS):
            columns.append(
                f"value_{sample_index}_{channel_index}"
            )

    return columns


SENSOR_COLUMNS = create_sensor_column_names()

FINAL_COLUMNS = (
    ["actual", "trial_number"]
    + SENSOR_COLUMNS
)


# ============================================================
# SAVE / VALIDATION
# ============================================================

def save_dataset(rows, filename):
    df = pd.DataFrame(
        rows,
        columns=FINAL_COLUMNS,
    )

    df.to_csv(
        filename,
        index=False,
    )

    return df


def validate_dataset(df):
    section("FINAL DATASET VALIDATION")

    print(f"Shape: {df.shape}")
    print()

    print("Class counts:")
    print(df["actual"].value_counts().sort_index())
    print()

    expected_rows = (
        len(NUMBERS)
        * TRIALS_PER_NUMBER
    )

    if df.shape != (
        expected_rows,
        2 + EXPECTED_VALUES,
    ):
        raise ValueError(
            f"Expected shape "
            f"({expected_rows}, {2 + EXPECTED_VALUES}), "
            f"got {df.shape}."
        )

    print(
        f"PASS: Dataset contains "
        f"{expected_rows} rows and "
        f"{2 + EXPECTED_VALUES} columns."
    )

    for number in NUMBERS:
        count = int(
            (df["actual"].astype(str) == number).sum()
        )

        if count != TRIALS_PER_NUMBER:
            raise ValueError(
                f"Number {number}: expected "
                f"{TRIALS_PER_NUMBER}, found {count}."
            )

        print(
            f"PASS: {number} = "
            f"{count}/{TRIALS_PER_NUMBER}"
        )

    sensor_matrix = df[SENSOR_COLUMNS]

    missing_values = int(
        sensor_matrix.isna().sum().sum()
    )

    print(
        f"Sensor columns: {len(SENSOR_COLUMNS)}"
    )
    print(
        f"Missing values: {missing_values}"
    )

    if missing_values != 0:
        raise ValueError(
            "Dataset contains missing sensor values."
        )

    values = sensor_matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Dataset contains NaN or infinite values."
        )

    print()
    print(
        "PASS: COMPLETE NUMBER 1-5 DATASET VALIDATED."
    )


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

def main():

    section("MPU6050 NUMBER 1-5 LIVE DATA COLLECTION")

    print("Collection plan:")
    print()
    print("Numbers: 1, 2, 3, 4, 5")
    print("Trials per number: 10")
    print("Total recordings: 50")
    print("Samples per recording: 100")
    print("MPU6050 channels: 6")
    print("Sensor values per recording: 600")
    print()
    print("The Python script will automatically send R.")
    print("Do NOT type R manually into Terminal.")
    print()

    print("IMPORTANT:")
    print("Keep the MPU6050 orientation consistent.")
    print("Use the same starting pose for every trial.")
    print("Draw each number using the same stroke convention.")
    print("Keep gesture size and speed reasonably consistent.")
    print()

    input("Press ENTER to begin...")

    # --------------------------------------------------------
    # SERIAL CONNECTION
    # --------------------------------------------------------

    section("SERIAL CONNECTION")

    port = choose_serial_port()

    print(f"Selected port: {port}")
    print(f"Baud rate: {BAUD_RATE}")
    print()
    print("Opening Arduino connection...")

    try:
        ser = serial.Serial(
            port,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT,
        )

    except serial.SerialException as exc:
        print()
        print(f"ERROR: Could not open serial port.")
        print(exc)
        sys.exit(1)

    print("Waiting 2 seconds for Arduino...")
    time.sleep(2.0)

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("PASS: Serial connection opened.")

    rows = []

    # --------------------------------------------------------
    # NUMBER COLLECTION
    # --------------------------------------------------------

    try:

        for number in NUMBERS:

            section(
                f"NUMBER {number} COLLECTION"
            )

            print(
                f"You will now record NUMBER {number} "
                f"{TRIALS_PER_NUMBER} times."
            )
            print()

            trial = 1

            while trial <= TRIALS_PER_NUMBER:

                print()
                print("-" * 78)
                print(
                    f"NUMBER {number} — "
                    f"TRIAL {trial}/{TRIALS_PER_NUMBER}"
                )
                print("-" * 78)
                print()

                print(
                    f"Prepare to AIR-DRAW the number: {number}"
                )
                print()
                print(
                    "Return the MPU6050 to the SAME starting "
                    "orientation before each trial."
                )

                input(
                    "\nPress ENTER when you are in the "
                    "starting position..."
                )

                print()
                print("Get ready...")
                time.sleep(0.5)

                for countdown in [3, 2, 1]:
                    print(countdown, flush=True)
                    time.sleep(0.65)

                print()
                print(
                    f">>> DRAW NUMBER {number} NOW <<<"
                )
                print(
                    "Arduino recording has been triggered "
                    "automatically."
                )
                print()

                recording = capture_recording(ser)

                if recording is None:

                    print()
                    print(
                        f"NUMBER {number} Trial {trial} "
                        f"FAILED."
                    )
                    print(
                        "This trial was NOT saved."
                    )
                    print(
                        "It will be repeated automatically."
                    )

                    input(
                        "\nPress ENTER when ready to retry..."
                    )

                    continue

                # Successful trial.
                row = (
                    [number, trial]
                    + recording.tolist()
                )

                rows.append(row)

                print()
                print(
                    f"PASS: NUMBER {number} "
                    f"Trial {trial} captured."
                )
                print(
                    f"Validated sensor values: "
                    f"{len(recording)}/{EXPECTED_VALUES}"
                )

                # Save progress after every successful trial.
                progress_df = save_dataset(
                    rows,
                    PROGRESS_FILE,
                )

                print(
                    f"Progress saved: {PROGRESS_FILE}"
                )

                completed = len(rows)
                total = (
                    len(NUMBERS)
                    * TRIALS_PER_NUMBER
                )

                print(
                    f"Overall progress: "
                    f"{completed}/{total}"
                )

                trial += 1

        # ----------------------------------------------------
        # FINAL SAVE
        # ----------------------------------------------------

        section("ALL NUMBER RECORDINGS CAPTURED")

        final_df = save_dataset(
            rows,
            OUTPUT_FILE,
        )

        validate_dataset(final_df)

        # Remove temporary progress file only after successful
        # final validation.
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

        section("NUMBER 1-5 COLLECTION COMPLETE")

        print("Saved:")
        print(f"    {OUTPUT_FILE}")
        print()
        print(f"Final shape: {final_df.shape}")
        print()
        print("Expected:")
        print("    50 recordings")
        print("    10 recordings each for 1, 2, 3, 4, 5")
        print("    602 columns")
        print("    2 metadata columns")
        print("    600 MPU6050 sensor values")
        print()
        print("IMPORTANT:")
        print(
            "Do NOT combine this dataset with A-L yet."
        )
        print(
            "Next step is to train and validate a "
            "SEPARATE number 1-5 model."
        )

    except KeyboardInterrupt:
        print()
        print()
        section("COLLECTION INTERRUPTED")

        if rows:
            save_dataset(
                rows,
                PROGRESS_FILE,
            )

            print(
                f"Partial progress preserved in:"
            )
            print(
                f"    {PROGRESS_FILE}"
            )

        else:
            print(
                "No completed trials were available to save."
            )

    finally:

        try:
            ser.close()
        except Exception:
            pass

        print()
        print("Serial connection closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
