"""
TARGETED DIGIT-6 REMEDIATION COLLECTION

Purpose:
    Collect 20 additional fresh training recordings of digit 6.

Why:
    Latest 105-model certification:
        6 -> 0/5 correct
        predicted as 9 three times
        predicted as 0 two times

These 20 recordings remain SEPARATE from:
    - original 50-recording dataset
    - previous 55-recording remediation dataset
    - certification/holdout data

Target:
    Digit 6 only
    20 fresh recordings

Each recording:
    100 samples
    6 MPU6050 channels

Arduino protocol:
    Python sends uppercase R

    Arduino sends:
        START
        100 rows of:
            ax,ay,az,gx,gy,gz
        END

Expected units:
    acceleration = m/s^2
    gyroscope = rad/s

NO unit conversion is applied.

The script:
    - saves after every successful trial
    - resumes automatically if interrupted
    - never modifies existing datasets
"""

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURATION
# ============================================================

DIGIT = "6"

TARGET_TRIALS = 20

NUM_SAMPLES = 100
NUM_CHANNELS = 6

BAUD_RATE = 115200

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12

OUTPUT_FILE = (
    "digit6_targeted_remediation_20_raw_data.csv"
)


# ============================================================
# SERIAL PORT
# ============================================================

def find_serial_ports():

    ports = []

    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]

    for pattern in patterns:

        ports.extend(
            glob.glob(pattern)
        )

    return sorted(
        set(ports)
    )


def choose_serial_port():

    ports = find_serial_ports()

    if not ports:

        raise RuntimeError(
            "\nNo Arduino serial port found.\n"
            "Run:\n"
            "    ls /dev/cu.*\n"
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

    print(
        "Detected ports:"
    )

    for i, port in enumerate(
        ports,
        start=1
    ):

        print(
            f"{i}. {port}"
        )

    while True:

        answer = input(
            "Choose port number: "
        ).strip()

        try:

            index = (
                int(answer) - 1
            )

            if (
                0
                <= index
                < len(ports)
            ):

                return ports[index]

        except ValueError:

            pass

        print(
            "Invalid selection."
        )


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
# ARDUINO COMMUNICATION
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

    # Current Arduino protocol:
    # uppercase R starts one recording.
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

        if len(parts) != 6:

            continue

        try:

            row = [
                float(value)
                for value in parts
            ]

        except ValueError:

            continue

        if not np.all(
            np.isfinite(row)
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
            f"got {recording.shape}"
        )

    print()
    print(
        f"Capture duration: "
        f"{elapsed:.3f} seconds"
    )

    print(
        "Samples:",
        recording.shape[0]
    )

    return recording


# ============================================================
# SENSOR CHECK
# ============================================================

def sensor_summary(recording):

    accel = recording[
        :,
        0:3
    ]

    magnitude = np.sqrt(
        np.sum(
            accel ** 2,
            axis=1
        )
    )

    return (
        float(
            np.mean(magnitude)
        ),

        float(
            np.median(magnitude)
        ),

        float(
            np.min(magnitude)
        ),

        float(
            np.max(magnitude)
        ),
    )


# ============================================================
# CSV FORMAT
# ============================================================

def make_row(
    trial_number,
    recording
):

    row = {
        "actual": DIGIT,
        "trial_number": trial_number,
    }

    for sample_index in range(
        NUM_SAMPLES
    ):

        for channel in range(
            NUM_CHANNELS
        ):

            column = (
                f"value_"
                f"{sample_index}_"
                f"{channel}"
            )

            row[column] = float(
                recording[
                    sample_index,
                    channel
                ]
            )

    return row


# ============================================================
# EXISTING DATA / RESUME
# ============================================================

def load_existing():

    path = Path(
        OUTPUT_FILE
    )

    if not path.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        path
    )

    print()
    print(
        "Existing digit-6 remediation "
        "file found."
    )

    print(
        "Existing rows:",
        len(df)
    )

    if len(df) > TARGET_TRIALS:

        raise RuntimeError(
            "Existing file contains more "
            "than 20 rows."
        )

    print(
        "Collection will resume."
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_rows(rows):

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_dataset(df):

    print()
    print("=" * 72)
    print(
        "DIGIT-6 REMEDIATION VALIDATION"
    )
    print("=" * 72)

    print(
        "Rows:",
        len(df)
    )

    labels = (
        df["actual"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.strip()
    )

    digit6_count = int(
        (
            labels == DIGIT
        ).sum()
    )

    print(
        "Digit 6 rows:",
        digit6_count
    )

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(
            "value_"
        )
    ]

    print(
        "Sensor columns:",
        len(sensor_columns)
    )

    if len(sensor_columns) != 600:

        print(
            "VALIDATION FAILED: "
            "expected 600 sensor columns."
        )

        return

    values = (
        df[
            sensor_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    missing = int(
        np.isnan(
            values
        ).sum()
    )

    finite = bool(
        np.all(
            np.isfinite(
                values
            )
        )
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "All finite:",
        finite
    )

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

    duplicate_count = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print(
        "Exact duplicate recordings:",
        duplicate_count
    )

    if (
        len(df) == TARGET_TRIALS
        and digit6_count == TARGET_TRIALS
        and len(sensor_columns) == 600
        and missing == 0
        and finite
        and duplicate_count == 0
    ):

        print()
        print(
            "DIGIT-6 REMEDIATION: PASS"
        )

    else:

        print()
        print(
            "DIGIT-6 REMEDIATION: "
            "CHECK REQUIRED"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "TARGETED DIGIT-6 "
        "REMEDIATION COLLECTION"
    )
    print("=" * 72)

    print()
    print(
        "Target:",
        TARGET_TRIALS,
        "fresh digit-6 recordings"
    )

    print()
    print(
        "Important:"
    )

    print(
        "Use your natural digit-6 gesture."
    )

    print(
        "Vary size, speed, and position "
        "slightly across trials."
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    existing_df = (
        load_existing()
    )

    rows = []

    if len(existing_df):

        rows = (
            existing_df
            .to_dict(
                orient="records"
            )
        )

    completed = len(
        rows
    )

    if completed >= TARGET_TRIALS:

        print()
        print(
            "All 20 recordings are "
            "already complete."
        )

        validate_dataset(
            pd.DataFrame(
                rows
            )
        )

        return

    port = choose_serial_port()

    print()
    print(
        "Opening:",
        port
    )

    ser = serial.Serial(
        port,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(
        2
    )

    ser.reset_input_buffer()

    print(
        "Serial connection ready."
    )

    try:

        for trial in range(
            completed + 1,
            TARGET_TRIALS + 1
        ):

            while True:

                print()
                print("=" * 72)

                print(
                    f"DIGIT 6 — "
                    f"Trial {trial}/"
                    f"{TARGET_TRIALS}"
                )

                print("=" * 72)

                answer = input(
                    "Press ENTER when ready "
                    "or q to quit: "
                ).strip().lower()

                if answer == "q":

                    raise KeyboardInterrupt

                countdown()

                try:

                    recording = (
                        capture_recording(
                            ser
                        )
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

                    retry = input(
                        "Press ENTER to retry "
                        "or q to quit: "
                    ).strip().lower()

                    if retry == "q":

                        raise KeyboardInterrupt

                    continue

                (
                    mean_mag,
                    median_mag,
                    min_mag,
                    max_mag,
                ) = sensor_summary(
                    recording
                )

                print()
                print(
                    "Acceleration magnitude:"
                )

                print(
                    "  mean:",
                    f"{mean_mag:.4f}"
                )

                print(
                    "  median:",
                    f"{median_mag:.4f}"
                )

                print(
                    "  min:",
                    f"{min_mag:.4f}"
                )

                print(
                    "  max:",
                    f"{max_mag:.4f}"
                )

                if not (
                    5.0
                    <= median_mag
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

                        print(
                            "Trial discarded."
                        )

                        continue

                rows.append(
                    make_row(
                        trial,
                        recording
                    )
                )

                df = save_rows(
                    rows
                )

                print()
                print(
                    "SAVED."
                )

                print(
                    "Current digit-6 rows:",
                    len(df)
                )

                print(
                    "File:",
                    OUTPUT_FILE
                )

                break

    except KeyboardInterrupt:

        print()
        print(
            "Collection stopped."
        )

    finally:

        ser.close()

        print(
            "Serial connection closed."
        )

    if rows:

        final_df = save_rows(
            rows
        )

        validate_dataset(
            final_df
        )

        print()
        print(
            "Saved digit-6 remediation:"
        )

        print(
            OUTPUT_FILE
        )


if __name__ == "__main__":

    main()
