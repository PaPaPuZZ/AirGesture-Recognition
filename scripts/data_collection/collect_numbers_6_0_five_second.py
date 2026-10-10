"""
NUMBERS 6-0 FRESH DATA COLLECTION
5-SECOND PYTHON RECORDING WINDOW

Digits:
    6, 7, 8, 9, 0

Collection:
    10 trials per digit
    50 trials total

Per trial:
    3-second countdown
    100 samples
    6 channels
    50 ms between accepted samples
    5.00-second recording window

Arduino protocol:
    Python sends:
        R

    Arduino sends:
        START
        sensor rows:
            ax,ay,az,gx,gy,gz
        END

Sensor units expected:
    acceleration = m/s^2
    gyro = rad/s

No unit conversion is performed.
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

SAMPLE_INTERVAL_SECONDS = 0.050
RECORDING_DURATION_SECONDS = 5.0

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
SERIAL_TIMEOUT_SECONDS = 1

DIGITS = [
    "6",
    "7",
    "8",
    "9",
    "0",
]

TRIALS_PER_DIGIT = 10

OUTPUT_FILE = (
    "numbers_6_0_5second_training_raw_data.csv"
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
            "Run:\n\n"
            "    ls /dev/cu.*\n"
        )

    print()
    print("=" * 70)
    print("SERIAL PORT")
    print("=" * 70)

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

            choice = int(answer) - 1

            if (
                0
                <= choice
                < len(ports)
            ):

                return ports[choice]

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
# WAIT FOR START
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


# ============================================================
# READ ONE VALID SENSOR ROW
# ============================================================

def read_sensor_row(
    ser
):

    while True:

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

            raise RuntimeError(
                "Arduino sent END before "
                "100 samples were collected."
            )

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

        return row


# ============================================================
# FIVE-SECOND RECORDING
# ============================================================

def capture_recording(
    ser
):

    ser.reset_input_buffer()

    # Begin Arduino recording.
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

    recording_start = (
        time.monotonic()
    )

    print()
    print(
        "Recording for 5.00 seconds..."
    )

    for sample_index in range(
        NUM_SAMPLES
    ):

        target_time = (
            recording_start
            + (
                sample_index
                * SAMPLE_INTERVAL_SECONDS
            )
        )

        # Wait until this sample's
        # 50 ms position.
        while (
            time.monotonic()
            < target_time
        ):

            time.sleep(
                0.001
            )

        row = read_sensor_row(
            ser
        )

        samples.append(
            row
        )

        # Optional visible progress.
        if (
            sample_index + 1
        ) % 20 == 0:

            elapsed = (
                time.monotonic()
                - recording_start
            )

            print(
                f"  {sample_index + 1}/"
                f"{NUM_SAMPLES} samples "
                f"| {elapsed:.2f} s"
            )

    # Make sure the full five-second
    # recording window has elapsed.
    target_end = (
        recording_start
        + RECORDING_DURATION_SECONDS
    )

    while (
        time.monotonic()
        < target_end
    ):

        time.sleep(
            0.001
        )

    recording_end = (
        time.monotonic()
    )

    elapsed = (
        recording_end
        - recording_start
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
            f"received {recording.shape}"
        )

    print()
    print(
        "Recording finished."
    )

    print(
        f"Actual Python recording duration: "
        f"{elapsed:.3f} seconds"
    )

    print(
        "Samples:",
        len(recording)
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
# BUILD CSV ROW
# ============================================================

def make_csv_row(
    digit,
    trial,
    recording
):

    row = {
        "actual": digit,
        "trial_number": trial,
    }

    for sample_index in range(
        NUM_SAMPLES
    ):

        for channel_index in range(
            NUM_CHANNELS
        ):

            column = (
                f"value_"
                f"{sample_index}_"
                f"{channel_index}"
            )

            row[column] = float(
                recording[
                    sample_index,
                    channel_index
                ]
            )

    return row


# ============================================================
# SAVE DATA
# ============================================================

def save_rows(
    rows
):

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return dataframe


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_dataset(
    dataframe
):

    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        len(dataframe)
    )

    labels = (
        dataframe["actual"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
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
            f"{digit}: {count}"
        )

    sensor_columns = [
        column
        for column
        in dataframe.columns
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
            "FAIL:"
            " expected 600 sensor columns."
        )

        return

    values = (
        dataframe[
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
        len(dataframe),
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

    correct_counts = all(

        int(
            (
                labels == digit
            ).sum()
        )
        == TRIALS_PER_DIGIT

        for digit in DIGITS
    )

    if (
        len(dataframe) == 50
        and missing == 0
        and finite
        and correct_counts
    ):

        print()
        print(
            "DATASET VALIDATION: PASS"
        )

    else:

        print()
        print(
            "DATASET VALIDATION: "
            "CHECK REQUIRED"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "NUMBERS 6-0 "
        "FIVE-SECOND COLLECTION"
    )
    print("=" * 70)

    print()
    print(
        "Starts at:"
    )

    print(
        "Digit 6 — Trial 1/10"
    )

    print()
    print(
        "Recording settings:"
    )

    print(
        "  Samples:",
        NUM_SAMPLES
    )

    print(
        "  Channels:",
        NUM_CHANNELS
    )

    print(
        "  Sample interval:",
        "50 ms"
    )

    print(
        "  Recording window:",
        "5.00 seconds"
    )

    print(
        "  Countdown:",
        "3 seconds"
    )

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if Path(
        OUTPUT_FILE
    ).exists():

        print()
        print(
            "ERROR:"
        )

        print(
            OUTPUT_FILE,
            "already exists."
        )

        print()
        print(
            "Rename the old file first "
            "if you want to restart."
        )

        return

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    port = choose_serial_port()

    print()
    print(
        "Opening:",
        port
    )

    ser = serial.Serial(
        port,
        BAUD_RATE,
        timeout=SERIAL_TIMEOUT_SECONDS
    )

    time.sleep(
        2
    )

    ser.reset_input_buffer()

    print(
        "Serial connection ready."
    )

    rows = []

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    try:

        for digit in DIGITS:

            print()
            print("=" * 70)
            print(
                f"DIGIT {digit}"
            )
            print("=" * 70)

            for trial in range(
                1,
                TRIALS_PER_DIGIT + 1
            ):

                while True:

                    print()
                    print(
                        f"Digit {digit} — "
                        f"Trial {trial}/"
                        f"{TRIALS_PER_DIGIT}"
                    )

                    command = input(
                        "Press ENTER when ready "
                        "or type q to quit: "
                    ).strip().lower()

                    if command == "q":

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
                            "WARNING:"
                        )

                        print(
                            "Acceleration magnitude "
                            "looks unusual."
                        )

                        answer = input(
                            "Save anyway? "
                            "(y/n): "
                        ).strip().lower()

                        if answer != "y":

                            print(
                                "Trial discarded."
                            )

                            continue

                    row = make_csv_row(
                        digit,
                        trial,
                        recording
                    )

                    rows.append(
                        row
                    )

                    dataframe = save_rows(
                        rows
                    )

                    print()
                    print(
                        "SAVED."
                    )

                    print(
                        "Current total rows:",
                        len(dataframe)
                    )

                    print(
                        "File:",
                        OUTPUT_FILE
                    )

                    break

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Collection stopped."
        )

        if rows:

            save_rows(
                rows
            )

            print(
                "Current progress saved."
            )

    finally:

        ser.close()

        print()
        print(
            "Serial connection closed."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if rows:

        final_dataframe = (
            save_rows(
                rows
            )
        )

        validate_dataset(
            final_dataframe
        )


if __name__ == "__main__":

    main()
