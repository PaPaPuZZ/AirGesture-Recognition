"""
TARGETED NUMBERS 6-0 REMEDIATION DATA COLLECTION

Purpose:
Collect additional training examples specifically for the
classes that performed poorly during live certification.

Target counts:
    6 -> 5 new examples
    7 -> 5 new examples
    8 -> 10 new examples
    9 -> 15 new examples
    0 -> 20 new examples

Total:
    55 fresh remediation recordings

Each recording:
    100 samples
    6 channels
    approximately 5 seconds
    current Arduino protocol:
        Python sends R
        Arduino sends START
        100 sensor rows
        Arduino sends END

Sensor units:
    acceleration = m/s^2
    gyro = rad/s

This script:
    - saves remediation data to a NEW CSV
    - does not modify the original 50-recording dataset
    - saves after every successful recording
    - can resume automatically if interrupted
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

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12

TARGET_COUNTS = {
    "6": 5,
    "7": 5,
    "8": 10,
    "9": 15,
    "0": 20,
}

DIGITS = [
    "6",
    "7",
    "8",
    "9",
    "0",
]

OUTPUT_FILE = (
    "numbers_6_0_targeted_remediation_raw_data.csv"
)


# ============================================================
# SERIAL PORT
# ============================================================

def find_serial_ports():

    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
    ]

    ports = []

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

            index = int(answer) - 1

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

def wait_for_start(
    ser
):

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


def capture_recording(
    ser
):

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
            "Timed out before END."
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
            f"({NUM_SAMPLES}, "
            f"{NUM_CHANNELS}), "
            f"got {recording.shape}"
        )

    print()
    print(
        f"Capture duration: "
        f"{elapsed:.3f} seconds"
    )

    return recording


# ============================================================
# SENSOR SUMMARY
# ============================================================

def sensor_summary(
    recording
):

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
# CSV ROW
# ============================================================

def make_row(
    digit,
    trial_number,
    recording
):

    row = {
        "actual": digit,
        "trial_number": trial_number,
    }

    for sample_index in range(
        NUM_SAMPLES
    ):

        for channel in range(
            NUM_CHANNELS
        ):

            key = (
                f"value_"
                f"{sample_index}_"
                f"{channel}"
            )

            row[key] = float(
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
        "Existing remediation file found."
    )

    print(
        "Rows:",
        len(df)
    )

    print()
    print(
        "Collection will resume."
    )

    return df


def normalized_labels(
    df
):

    if (
        len(df) == 0
        or "actual"
        not in df.columns
    ):

        return pd.Series(
            dtype=str
        )

    return (
        df["actual"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.strip()
    )


def completed_count(
    df,
    digit
):

    labels = normalized_labels(
        df
    )

    if len(labels) == 0:

        return 0

    return int(
        (
            labels == digit
        ).sum()
    )


# ============================================================
# SAVE
# ============================================================

def save_rows(
    rows
):

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    df
):

    print()
    print("=" * 72)
    print(
        "REMEDIATION DATASET VALIDATION"
    )
    print("=" * 72)

    print(
        "Rows:",
        len(df)
    )

    labels = normalized_labels(
        df
    )

    print()
    print(
        "Class counts:"
    )

    for digit in DIGITS:

        count = int(
            (
                labels == digit
            ).sum()
        )

        print(
            f"{digit}: "
            f"{count}/"
            f"{TARGET_COUNTS[digit]}"
        )

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

    if len(
        sensor_columns
    ) != 600:

        print(
            "FAIL: expected 600 "
            "sensor columns."
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

    accel = recordings[
        :,
        :,
        0:3
    ]

    magnitude = np.sqrt(
        np.sum(
            accel ** 2,
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

    duplicates = int(
        pd.DataFrame(
            values
        )
        .duplicated()
        .sum()
    )

    print(
        "Exact duplicate recordings:",
        duplicates
    )

    expected_total = sum(
        TARGET_COUNTS.values()
    )

    correct_counts = all(

        int(
            (
                labels == digit
            ).sum()
        )
        == TARGET_COUNTS[
            digit
        ]

        for digit in DIGITS
    )

    if (
        len(df) == expected_total
        and len(sensor_columns) == 600
        and missing == 0
        and finite
        and correct_counts
    ):

        print()
        print(
            "REMEDIATION DATASET: PASS"
        )

    else:

        print()
        print(
            "REMEDIATION DATASET: "
            "INCOMPLETE / CHECK REQUIRED"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 6-0 "
        "TARGETED REMEDIATION COLLECTION"
    )
    print("=" * 72)

    print()
    print(
        "Target additional recordings:"
    )

    for digit in DIGITS:

        print(
            f"  {digit}: "
            f"{TARGET_COUNTS[digit]}"
        )

    print()
    print(
        "Total:",
        sum(
            TARGET_COUNTS.values()
        )
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

    if len(
        existing_df
    ):

        rows = (
            existing_df
            .to_dict(
                orient="records"
            )
        )

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

            target = (
                TARGET_COUNTS[
                    digit
                ]
            )

            if already_done >= target:

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
                f"Already completed: "
                f"{already_done}/"
                f"{target}"
            )

            for trial in range(
                already_done + 1,
                target + 1
            ):

                while True:

                    print()
                    print(
                        f"Digit {digit} — "
                        f"Remediation trial "
                        f"{trial}/{target}"
                    )

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
                            "Trial NOT saved."
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
                            "WARNING:"
                        )

                        print(
                            "Acceleration magnitude "
                            "looks unusual."
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
                            digit,
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
                        "Total remediation rows:",
                        len(df)
                    )

                    break

    except KeyboardInterrupt:

        print()
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
            "Saved remediation dataset:"
        )

        print(
            OUTPUT_FILE
        )


if __name__ == "__main__":

    main()
