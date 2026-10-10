"""
UNIFIED NUMBERS 0-9 LIVE CERTIFICATION

Model:
    mpu6050_numbers_0_9_best_model.pkl

Fresh holdout:
    3 trials per digit
    10 digits
    30 total trials

Raw input:
    100 samples x 6 channels

Feature extraction:
    4 temporal segments
    x 6 channels
    x 9 features
    = 216 total features

Feature order:
    mean
    std
    min
    max
    range
    rms
    avg_abs_change
    max_abs_change
    total_movement

No retraining occurs.
Training files and saved model are not modified.
"""

import glob
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "mpu6050_numbers_0_9_best_model.pkl"

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

EXPECTED_FEATURES = 216

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

TRIALS_PER_DIGIT = 3

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_segment_features(segment):

    segment = np.asarray(
        segment,
        dtype=float
    )

    if segment.shape != (
        SAMPLES_PER_SEGMENT,
        NUM_CHANNELS
    ):

        raise ValueError(
            f"Expected segment shape "
            f"({SAMPLES_PER_SEGMENT}, {NUM_CHANNELS}), "
            f"got {segment.shape}"
        )

    features = []

    for channel in range(
        NUM_CHANNELS
    ):

        x = segment[:, channel]

        dx = np.diff(x)

        features.extend(
            [
                np.mean(x),
                np.std(x),
                np.min(x),
                np.max(x),
                np.max(x) - np.min(x),
                np.sqrt(
                    np.mean(
                        x ** 2
                    )
                ),
                np.mean(
                    np.abs(dx)
                ),
                np.max(
                    np.abs(dx)
                ),
                np.sum(
                    np.abs(dx)
                ),
            ]
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != 54:

        raise RuntimeError(
            f"Expected 54 segment features, "
            f"got {features.size}"
        )

    return features


def extract_temporal_features(recording):

    recording = np.asarray(
        recording,
        dtype=float
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS
    ):

        raise ValueError(
            f"Expected recording shape "
            f"({NUM_SAMPLES}, {NUM_CHANNELS}), "
            f"got {recording.shape}"
        )

    features = []

    for segment_index in range(
        NUM_SEGMENTS
    ):

        start = (
            segment_index
            * SAMPLES_PER_SEGMENT
        )

        end = (
            start
            + SAMPLES_PER_SEGMENT
        )

        segment = recording[
            start:end,
            :
        ]

        features.extend(
            extract_segment_features(
                segment
            )
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {features.size}"
        )

    if not np.all(
        np.isfinite(
            features
        )
    ):

        raise RuntimeError(
            "Feature vector contains "
            "NaN or infinity."
        )

    return features


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    path = Path(
        MODEL_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{MODEL_FILE}"
        )

    with open(
        path,
        "rb"
    ) as handle:

        saved = pickle.load(
            handle
        )

    if not isinstance(
        saved,
        dict
    ):

        raise TypeError(
            "Expected saved model dictionary."
        )

    model = saved[
        "model"
    ]

    print()
    print("=" * 72)
    print("MODEL VERIFICATION")
    print("=" * 72)

    print(
        "Model name:",
        saved.get(
            "model_name"
        )
    )

    print(
        "Training rows:",
        saved.get(
            "training_rows"
        )
    )

    print(
        "Classes:",
        saved.get(
            "classes"
        )
    )

    print(
        "Expected features:",
        saved.get(
            "expected_features"
        )
    )

    print(
        "Feature definition:",
        saved.get(
            "feature_definition"
        )
    )

    if int(
        saved[
            "expected_features"
        ]
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "Model does not expect "
            "216 features."
        )

    expected_definition = [
        "mean",
        "std",
        "min",
        "max",
        "range",
        "rms",
        "avg_abs_change",
        "max_abs_change",
        "total_movement",
    ]

    if list(
        saved[
            "feature_definition"
        ]
    ) != expected_definition:

        raise RuntimeError(
            "Feature definition mismatch."
        )

    if hasattr(
        model,
        "n_features_in_"
    ):

        print(
            "Model n_features_in_:",
            model.n_features_in_
        )

        if (
            model.n_features_in_
            != EXPECTED_FEATURES
        ):

            raise RuntimeError(
                "Model input dimension "
                "does not equal 216."
            )

    print()
    print(
        "MODEL VERIFICATION: PASS"
    )

    return (
        saved,
        model
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
            glob.glob(
                pattern
            )
        )

    return sorted(
        set(
            ports
        )
    )


def choose_serial_port():

    ports = find_serial_ports()

    if not ports:

        raise RuntimeError(
            "No Arduino serial port found."
        )

    if len(ports) == 1:

        print(
            "Using serial port:",
            ports[0]
        )

        return ports[0]

    print()
    print(
        "Detected serial ports:"
    )

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
# SENSOR SUMMARY
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
            np.mean(
                magnitude
            )
        ),
        float(
            np.median(
                magnitude
            )
        ),
        float(
            np.min(
                magnitude
            )
        ),
        float(
            np.max(
                magnitude
            )
        ),
    )


# ============================================================
# RAW DATA ROW
# ============================================================

def make_raw_row(
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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "FULL NUMBERS 0-9 "
        "LIVE CERTIFICATION"
    )
    print("=" * 72)

    print()
    print(
        "Fresh holdout:"
    )

    print(
        "3 trials per digit"
    )

    print(
        "30 total trials"
    )

    saved, model = (
        load_model()
    )

    model_classes = [
        str(value)
        for value
        in model.classes_
    ]

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

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    raw_file = (
        "numbers_0_9_certification_raw_"
        + timestamp
        + ".csv"
    )

    results_file = (
        "numbers_0_9_certification_results_"
        + timestamp
        + ".csv"
    )

    summary_file = (
        "numbers_0_9_certification_summary_"
        + timestamp
        + ".txt"
    )

    raw_rows = []
    result_rows = []

    total_correct = 0

    try:

        for digit in DIGITS:

            print()
            print("=" * 72)
            print(
                f"DIGIT {digit}"
            )
            print("=" * 72)

            for trial in range(
                1,
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

                    features = (
                        extract_temporal_features(
                            recording
                        )
                    )

                    X = features.reshape(
                        1,
                        -1
                    )

                    prediction = str(
                        model.predict(
                            X
                        )[0]
                    )

                    probabilities = (
                        model.predict_proba(
                            X
                        )[0]
                    )

                    probability_map = {
                        str(label): float(prob)
                        for label, prob
                        in zip(
                            model.classes_,
                            probabilities
                        )
                    }

                    confidence = float(
                        probability_map[
                            prediction
                        ]
                    )

                    correct = (
                        prediction
                        == digit
                    )

                    if correct:

                        total_correct += 1

                    print()
                    print(
                        "Prediction:",
                        prediction
                    )

                    print(
                        "Confidence:",
                        f"{confidence * 100:.2f}%"
                    )

                    print(
                        "Result:",
                        "CORRECT"
                        if correct
                        else "WRONG"
                    )

                    print(
                        "Top probabilities:"
                    )

                    ranked = sorted(
                        probability_map.items(),
                        key=lambda item:
                            item[1],
                        reverse=True
                    )

                    for label, probability in (
                        ranked[:3]
                    ):

                        print(
                            f"  {label}: "
                            f"{probability * 100:.2f}%"
                        )

                    raw_rows.append(
                        make_raw_row(
                            digit,
                            trial,
                            recording
                        )
                    )

                    result = {
                        "actual":
                            digit,

                        "trial_number":
                            trial,

                        "prediction":
                            prediction,

                        "correct":
                            correct,

                        "confidence":
                            confidence,

                        "accel_magnitude_mean":
                            mean_mag,

                        "accel_magnitude_median":
                            median_mag,

                        "feature_count":
                            features.size,
                    }

                    for label in model_classes:

                        result[
                            f"prob_{label}"
                        ] = (
                            probability_map[
                                label
                            ]
                        )

                    result_rows.append(
                        result
                    )

                    print()
                    print(
                        "Trial saved."
                    )

                    break

    except KeyboardInterrupt:

        print()
        print(
            "Certification stopped."
        )

    finally:

        ser.close()

        print()
        print(
            "Serial connection closed."
        )

    if not result_rows:

        print(
            "No completed trials."
        )

        return

    raw_df = pd.DataFrame(
        raw_rows
    )

    results_df = pd.DataFrame(
        result_rows
    )

    raw_df.to_csv(
        raw_file,
        index=False
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    total_trials = len(
        results_df
    )

    accuracy = (
        total_correct
        / total_trials
    )

    print()
    print("=" * 72)
    print(
        "FINAL CERTIFICATION RESULTS"
    )
    print("=" * 72)

    print()
    print(
        f"Overall: "
        f"{total_correct}/"
        f"{total_trials}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        "Per-digit results:"
    )

    summary = []

    summary.append(
        "FULL NUMBERS 0-9 "
        "LIVE CERTIFICATION\n"
    )

    summary.append(
        "=" * 60
        + "\n"
    )

    summary.append(
        f"Model: "
        f"{saved['model_name']}\n"
    )

    summary.append(
        f"Training rows: "
        f"{saved['training_rows']}\n"
    )

    summary.append(
        f"Overall: "
        f"{total_correct}/"
        f"{total_trials}\n"
    )

    summary.append(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    for digit in DIGITS:

        subset = results_df[
            results_df[
                "actual"
            ].astype(str)
            == digit
        ]

        if len(subset) == 0:

            continue

        correct_count = int(
            subset[
                "correct"
            ].sum()
        )

        digit_accuracy = (
            correct_count
            / len(subset)
        )

        mean_confidence = float(
            subset[
                "confidence"
            ].mean()
        )

        line = (
            f"{digit}: "
            f"{correct_count}/"
            f"{len(subset)} "
            f"({digit_accuracy * 100:.1f}%) "
            f"| mean confidence "
            f"{mean_confidence * 100:.1f}%"
        )

        print(
            line
        )

        summary.append(
            line
            + "\n"
        )

    summary.append(
        "\nRaw data: "
        + raw_file
        + "\n"
    )

    summary.append(
        "Results: "
        + results_file
        + "\n"
    )

    Path(
        summary_file
    ).write_text(
        "".join(
            summary
        )
    )

    print()
    print(
        "Saved raw data:"
    )

    print(
        raw_file
    )

    print()
    print(
        "Saved results:"
    )

    print(
        results_file
    )

    print()
    print(
        "Saved summary:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Training data and model "
        "were NOT modified."
    )


if __name__ == "__main__":

    main()
