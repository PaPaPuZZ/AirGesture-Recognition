"""
AUTHORITATIVE NUMBERS 1-5 LIVE CERTIFICATION

Uses the confirmed saved model:
    mpu6050_numbers_1_5_best_model.pkl

Confirmed model:
    RandomForestClassifier
    classes = 1, 2, 3, 4, 5
    expected features = 216

Confirmed feature extraction:
    4 temporal segments
    25 samples per segment
    6 channels
    9 features per channel

Features, in exact order:
    1. mean
    2. std
    3. min
    4. max
    5. range
    6. RMS
    7. average absolute change
    8. maximum absolute change
    9. total movement

Arduino protocol:
    Python sends: R
    Arduino sends:
        START
        100 rows of 6 comma-separated values
        END

Current Arduino output is assumed to already be:
    acceleration = m/s^2
    gyro = rad/s

DO NOT apply another unit conversion.

Certification:
    5 trials each for digits 1-5
    25 fresh trials total

This script DOES NOT retrain or overwrite the saved model.
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

MODEL_FILE = "mpu6050_numbers_1_5_best_model.pkl"

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

FEATURES_PER_CHANNEL_PER_SEGMENT = 9
EXPECTED_FEATURES = 216

DIGITS = ["1", "2", "3", "4", "5"]

TRIALS_PER_DIGIT = 5

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_segment_features(segment):
    """
    Extract the exact 54 features used by training.

    segment shape:
        (25, 6)

    For every channel:
        mean
        std
        min
        max
        range
        rms
        avg_abs_change
        max_abs_change
        total_movement

    6 channels x 9 features = 54
    """

    segment = np.asarray(
        segment,
        dtype=float
    )

    if segment.shape != (
        SAMPLES_PER_SEGMENT,
        NUM_CHANNELS,
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

        mean = np.mean(x)

        std = np.std(x)

        minimum = np.min(x)

        maximum = np.max(x)

        value_range = (
            maximum - minimum
        )

        rms = np.sqrt(
            np.mean(
                x ** 2
            )
        )

        avg_abs_change = np.mean(
            np.abs(dx)
        )

        max_abs_change = np.max(
            np.abs(dx)
        )

        total_movement = np.sum(
            np.abs(dx)
        )

        features.extend(
            [
                mean,
                std,
                minimum,
                maximum,
                value_range,
                rms,
                avg_abs_change,
                max_abs_change,
                total_movement,
            ]
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != 54:
        raise RuntimeError(
            f"Segment produced "
            f"{features.size} features, "
            f"expected 54."
        )

    return features


def extract_temporal_features(
    recording
):
    """
    Convert one 100x6 recording into
    the exact 216-feature training vector.

    Ordering:
        Segment 1
            channel 0 -> 9 features
            channel 1 -> 9 features
            ...
            channel 5 -> 9 features

        Segment 2
            same ordering

        Segment 3
            same ordering

        Segment 4
            same ordering
    """

    recording = np.asarray(
        recording,
        dtype=float
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS,
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

        segment_features = (
            extract_segment_features(
                segment
            )
        )

        features.extend(
            segment_features
        )

    features = np.asarray(
        features,
        dtype=float
    )

    if features.size != (
        EXPECTED_FEATURES
    ):
        raise RuntimeError(
            f"Extractor produced "
            f"{features.size} features; "
            f"expected {EXPECTED_FEATURES}."
        )

    if not np.all(
        np.isfinite(features)
    ):
        raise RuntimeError(
            "Feature vector contains "
            "NaN or infinity."
        )

    return features


# ============================================================
# MODEL
# ============================================================

def load_model():
    """
    Load the saved dictionary and
    verify that it matches the pipeline
    we just confirmed.
    """

    path = Path(
        MODEL_FILE
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_FILE}"
        )

    with open(
        path,
        "rb"
    ) as f:

        saved = pickle.load(f)

    if not isinstance(
        saved,
        dict
    ):
        raise TypeError(
            "Saved model file is not "
            "the expected dictionary."
        )

    required_keys = [
        "model",
        "classes",
        "num_samples",
        "num_channels",
        "num_segments",
        "samples_per_segment",
        "features_per_channel_per_segment",
        "expected_features",
        "feature_definition",
    ]

    missing = [
        key
        for key in required_keys
        if key not in saved
    ]

    if missing:
        raise KeyError(
            "Saved model is missing keys: "
            + str(missing)
        )

    model = saved["model"]

    print()
    print("=" * 72)
    print("SAVED MODEL VERIFICATION")
    print("=" * 72)

    print(
        "Model name:",
        saved.get(
            "model_name",
            type(model).__name__
        )
    )

    print(
        "Classes:",
        saved["classes"]
    )

    print(
        "Samples:",
        saved["num_samples"]
    )

    print(
        "Channels:",
        saved["num_channels"]
    )

    print(
        "Segments:",
        saved["num_segments"]
    )

    print(
        "Samples per segment:",
        saved[
            "samples_per_segment"
        ]
    )

    print(
        "Features/channel/segment:",
        saved[
            "features_per_channel_per_segment"
        ]
    )

    print(
        "Expected features:",
        saved["expected_features"]
    )

    print(
        "Feature definition:",
        saved[
            "feature_definition"
        ]
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
                "Saved model does not "
                "expect 216 features."
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
            "Saved feature_definition "
            "does not match this script."
        )

    if int(
        saved[
            "expected_features"
        ]
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "Saved metadata expected_features "
            "does not equal 216."
        )

    print()
    print(
        "MODEL / FEATURE METADATA: PASS"
    )

    return saved, model


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
            "No Arduino serial port found.\n"
            "Run:\n"
            "    ls /dev/cu.*"
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

        choice = input(
            "Choose port number: "
        ).strip()

        try:

            index = int(choice) - 1

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
# ARDUINO CAPTURE
# ============================================================

def wait_for_start(
    ser
):

    deadline = (
        time.time()
        + START_TIMEOUT_SECONDS
    )

    while (
        time.time()
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

    # Remove old buffered text.
    ser.reset_input_buffer()

    # Current Arduino protocol
    # requires uppercase R.
    ser.write(
        b"R"
    )

    ser.flush()

    if not wait_for_start(
        ser
    ):

        raise TimeoutError(
            "Did not receive START "
            "from Arduino."
        )

    samples = []

    deadline = (
        time.time()
        + RECORDING_TIMEOUT_SECONDS
    )

    saw_end = False

    while (
        time.time()
        < deadline
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

            values = [
                float(x)
                for x in parts
            ]

        except ValueError:
            continue

        if not np.all(
            np.isfinite(values)
        ):
            continue

        samples.append(
            values
        )

    if not saw_end:

        raise TimeoutError(
            "Recording timed out "
            "before END."
        )

    recording = np.asarray(
        samples,
        dtype=float
    )

    if recording.shape != (
        NUM_SAMPLES,
        NUM_CHANNELS,
    ):

        raise RuntimeError(
            f"Expected exactly "
            f"{NUM_SAMPLES} x "
            f"{NUM_CHANNELS}, "
            f"received "
            f"{recording.shape}."
        )

    return recording


# ============================================================
# SENSOR SANITY CHECK
# ============================================================

def sensor_summary(
    recording
):

    accel = recording[
        :,
        0:3
    ]

    accel_magnitude = np.sqrt(
        np.sum(
            accel ** 2,
            axis=1
        )
    )

    mean_mag = float(
        np.mean(
            accel_magnitude
        )
    )

    median_mag = float(
        np.median(
            accel_magnitude
        )
    )

    return (
        mean_mag,
        median_mag
    )


# ============================================================
# SAVE RAW DATA
# ============================================================

def raw_row(
    actual,
    trial_number,
    recording
):

    row = {
        "actual": actual,
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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 1-5 AUTHORITATIVE "
        "LIVE CERTIFICATION"
    )
    print("=" * 72)

    print()
    print(
        "Fresh holdout:"
    )

    print(
        "5 digits x "
        f"{TRIALS_PER_DIGIT} trials "
        "= 25 trials"
    )

    print()
    print(
        "No retraining will occur."
    )

    print(
        "Existing model and training "
        "files will not be overwritten."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    saved, model = load_model()

    classes = [
        str(x)
        for x in model.classes_
    ]

    # --------------------------------------------------------
    # Serial connection
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
        timeout=1
    )

    # Arduino may reset when the
    # serial connection opens.
    time.sleep(2.0)

    ser.reset_input_buffer()

    print(
        "Serial connection ready."
    )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_filename = (
        "numbers_1_5_authoritative_"
        "certification_raw_"
        + timestamp
        + ".csv"
    )

    results_filename = (
        "numbers_1_5_authoritative_"
        "certification_results_"
        + timestamp
        + ".csv"
    )

    summary_filename = (
        "numbers_1_5_authoritative_"
        "certification_summary_"
        + timestamp
        + ".txt"
    )

    raw_rows = []

    result_rows = []

    total_correct = 0

    # --------------------------------------------------------
    # Certification
    # --------------------------------------------------------

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

                print()
                print(
                    f"Digit {digit} "
                    f"— Trial "
                    f"{trial}/"
                    f"{TRIALS_PER_DIGIT}"
                )

                input(
                    "Press ENTER when ready..."
                )

                print(
                    "Recording..."
                )

                recording = (
                    capture_recording(
                        ser
                    )
                )

                mean_mag, median_mag = (
                    sensor_summary(
                        recording
                    )
                )

                print(
                    "Samples:",
                    recording.shape[0]
                )

                print(
                    "Accel magnitude mean:",
                    f"{mean_mag:.4f}",
                    "m/s^2"
                )

                print(
                    "Accel magnitude median:",
                    f"{median_mag:.4f}",
                    "m/s^2"
                )

                # Typical stationary gravity
                # magnitude should be around
                # 9.8 m/s^2.
                if not (
                    5.0
                    <= median_mag
                    <= 15.0
                ):

                    print(
                        "WARNING: acceleration "
                        "magnitude looks unusual."
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

                confidence = (
                    probability_map[
                        prediction
                    ]
                )

                correct = (
                    prediction == digit
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
                    "Probabilities:"
                )

                for label in classes:

                    print(
                        f"  {label}: "
                        f"{probability_map[label] * 100:.2f}%"
                    )

                raw_rows.append(
                    raw_row(
                        digit,
                        trial,
                        recording
                    )
                )

                result = {
                    "actual": digit,
                    "trial_number": trial,
                    "prediction": prediction,
                    "correct": correct,
                    "confidence": confidence,
                    "accel_magnitude_mean": mean_mag,
                    "accel_magnitude_median": median_mag,
                    "feature_count": features.size,
                }

                for label in classes:

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

    finally:

        ser.close()

        print()
        print(
            "Serial connection closed."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    raw_df = pd.DataFrame(
        raw_rows
    )

    results_df = pd.DataFrame(
        result_rows
    )

    raw_df.to_csv(
        raw_filename,
        index=False
    )

    results_df.to_csv(
        results_filename,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    total_trials = len(
        results_df
    )

    accuracy = (
        total_correct
        / total_trials
        if total_trials
        else 0.0
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

    summary_lines = []

    summary_lines.append(
        "NUMBERS 1-5 AUTHORITATIVE "
        "CERTIFICATION\n"
    )

    summary_lines.append(
        "=" * 60
        + "\n"
    )

    summary_lines.append(
        f"Model: "
        f"{saved.get('model_name', type(model).__name__)}\n"
    )

    summary_lines.append(
        "Feature pipeline: "
        "4 segments x 6 channels x "
        "9 features = 216\n"
    )

    summary_lines.append(
        f"Overall: "
        f"{total_correct}/"
        f"{total_trials}\n"
    )

    summary_lines.append(
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

        print(line)

        summary_lines.append(
            line + "\n"
        )

    summary_lines.append(
        "\nRaw data: "
        + raw_filename
        + "\n"
    )

    summary_lines.append(
        "Results: "
        + results_filename
        + "\n"
    )

    Path(
        summary_filename
    ).write_text(
        "".join(
            summary_lines
        )
    )

    print()
    print(
        "Saved raw data:"
    )

    print(
        raw_filename
    )

    print()
    print(
        "Saved results:"
    )

    print(
        results_filename
    )

    print()
    print(
        "Saved summary:"
    )

    print(
        summary_filename
    )

    print()
    print(
        "Existing training data and "
        "model were NOT modified."
    )


if __name__ == "__main__":
    main()
