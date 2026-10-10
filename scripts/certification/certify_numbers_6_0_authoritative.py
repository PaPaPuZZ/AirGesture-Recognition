"""
AUTHORITATIVE NUMBERS 6-0 LIVE CERTIFICATION

Uses:
    mpu6050_numbers_6_0_best_model.pkl

Expected saved model:
    Best candidate from 6-0 training
    Currently: RBF SVM pipeline

Classes:
    0, 6, 7, 8, 9

Fresh holdout:
    5 trials per digit
    25 trials total

Raw recording:
    100 samples
    6 channels

Feature extraction:
    4 temporal segments
    25 samples per segment
    6 channels
    9 features per channel

Total:
    216 features

Feature order per channel per segment:
    mean
    std
    min
    max
    range
    rms
    avg_abs_change
    max_abs_change
    total_movement

Arduino protocol:
    Python sends uppercase R
    Arduino sends:
        START
        sensor data
        END

Expected sensor units:
    acceleration = m/s^2
    gyro = rad/s

No unit conversion is applied.

This script:
    - does NOT retrain the model
    - does NOT modify training data
    - saves fresh raw holdout data
    - saves trial-level predictions
    - saves a final summary
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

MODEL_FILE = (
    "mpu6050_numbers_6_0_best_model.pkl"
)

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

FEATURES_PER_CHANNEL_PER_SEGMENT = 9

EXPECTED_FEATURES = (
    NUM_SEGMENTS
    * NUM_CHANNELS
    * FEATURES_PER_CHANNEL_PER_SEGMENT
)

DIGITS = [
    "6",
    "7",
    "8",
    "9",
    "0",
]

TRIALS_PER_DIGIT = 5

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8

SERIAL_TIMEOUT_SECONDS = 1


# ============================================================
# AUTHORITATIVE FEATURE EXTRACTION
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
            f"({SAMPLES_PER_SEGMENT}, "
            f"{NUM_CHANNELS}), "
            f"got {segment.shape}"
        )

    features = []

    for channel in range(
        NUM_CHANNELS
    ):

        x = segment[
            :,
            channel
        ]

        dx = np.diff(
            x
        )

        features.extend(
            [
                np.mean(x),

                np.std(x),

                np.min(x),

                np.max(x),

                np.max(x)
                - np.min(x),

                np.sqrt(
                    np.mean(
                        x ** 2
                    )
                ),

                np.mean(
                    np.abs(
                        dx
                    )
                ),

                np.max(
                    np.abs(
                        dx
                    )
                ),

                np.sum(
                    np.abs(
                        dx
                    )
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
            f"got {features.size}."
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
            f"({NUM_SAMPLES}, "
            f"{NUM_CHANNELS}), "
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
            f"Expected "
            f"{EXPECTED_FEATURES} features, "
            f"got {features.size}."
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
# LOAD / VERIFY SAVED MODEL
# ============================================================

def load_saved_model():

    path = Path(
        MODEL_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find "
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
            "Saved model file is not "
            "the expected dictionary."
        )

    required = [
        "model",
        "model_name",
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
        for key in required
        if key not in saved
    ]

    if missing:

        raise RuntimeError(
            "Saved model is missing: "
            + str(missing)
        )

    model = saved[
        "model"
    ]

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

    print()
    print("=" * 72)
    print("MODEL VERIFICATION")
    print("=" * 72)

    print(
        "Model name:",
        saved[
            "model_name"
        ]
    )

    print(
        "Classes:",
        saved[
            "classes"
        ]
    )

    print(
        "Samples:",
        saved[
            "num_samples"
        ]
    )

    print(
        "Channels:",
        saved[
            "num_channels"
        ]
    )

    print(
        "Segments:",
        saved[
            "num_segments"
        ]
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
        saved[
            "expected_features"
        ]
    )

    print(
        "Feature definition:",
        saved[
            "feature_definition"
        ]
    )

    if int(
        saved[
            "num_samples"
        ]
    ) != NUM_SAMPLES:

        raise RuntimeError(
            "num_samples mismatch."
        )

    if int(
        saved[
            "num_channels"
        ]
    ) != NUM_CHANNELS:

        raise RuntimeError(
            "num_channels mismatch."
        )

    if int(
        saved[
            "num_segments"
        ]
    ) != NUM_SEGMENTS:

        raise RuntimeError(
            "num_segments mismatch."
        )

    if int(
        saved[
            "samples_per_segment"
        ]
    ) != SAMPLES_PER_SEGMENT:

        raise RuntimeError(
            "samples_per_segment mismatch."
        )

    if int(
        saved[
            "expected_features"
        ]
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "expected_features mismatch."
        )

    if list(
        saved[
            "feature_definition"
        ]
    ) != expected_definition:

        raise RuntimeError(
            "feature_definition mismatch."
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
                "Model does not expect "
                "216 features."
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
# SERIAL PORT
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
        set(ports)
    )


def choose_serial_port():

    ports = find_serial_ports()

    if not ports:

        raise RuntimeError(
            "\nNo Arduino serial port found.\n"
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

            selection = (
                int(answer)
                - 1
            )

            if (
                0
                <= selection
                < len(ports)
            ):

                return ports[
                    selection
                ]

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
# ARDUINO CAPTURE
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

    while True:

        raw = ser.readline()

        if not raw:

            if (
                time.monotonic()
                - start_time
            ) > 12:

                raise TimeoutError(
                    "Timed out during "
                    "recording."
                )

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

        raise RuntimeError(
            "END was not received."
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
            f"got "
            f"{recording.shape}"
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
# RAW CSV FORMAT
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
# MAIN CERTIFICATION
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "NUMBERS 6-0 "
        "AUTHORITATIVE CERTIFICATION"
    )
    print("=" * 72)

    print()
    print(
        "Fresh holdout:"
    )

    print(
        "5 trials each for "
        "6, 7, 8, 9, 0"
    )

    print(
        "25 total trials"
    )

    print()
    print(
        "No retraining will occur."
    )

    saved, model = (
        load_saved_model()
    )

    model_classes = [
        str(value)
        for value
        in model.classes_
    ]

    print()
    print(
        "Model classes:",
        model_classes
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
        timeout=SERIAL_TIMEOUT_SECONDS
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
        "numbers_6_0_authoritative_"
        "certification_raw_"
        + timestamp
        + ".csv"
    )

    results_file = (
        "numbers_6_0_authoritative_"
        "certification_results_"
        + timestamp
        + ".csv"
    )

    summary_file = (
        "numbers_6_0_authoritative_"
        "certification_summary_"
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

                        accept = input(
                            "Use trial anyway? "
                            "(y/n): "
                        ).strip().lower()

                        if accept != "y":

                            print(
                                "Trial discarded."
                            )

                            continue

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

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

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

                    else:

                        probability_map = {
                            label: np.nan
                            for label
                            in model_classes
                        }

                        confidence = np.nan

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

                    if np.isfinite(
                        confidence
                    ):

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

                    if (
                        probability_map
                        and np.isfinite(
                            list(
                                probability_map.values()
                            )[0]
                        )
                    ):

                        print(
                            "Probabilities:"
                        )

                        for label in model_classes:

                            value = (
                                probability_map[
                                    label
                                ]
                            )

                            print(
                                f"  {label}: "
                                f"{value * 100:.2f}%"
                            )

                    raw_rows.append(
                        make_raw_row(
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

    overall_accuracy = (
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
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print(
        "Per-digit results:"
    )

    summary_lines = []

    summary_lines.append(
        "NUMBERS 6-0 AUTHORITATIVE "
        "CERTIFICATION\n"
    )

    summary_lines.append(
        "=" * 60
        + "\n"
    )

    summary_lines.append(
        f"Model: "
        f"{saved['model_name']}\n"
    )

    summary_lines.append(
        "Feature pipeline: "
        "4 segments x 6 channels "
        "x 9 features = 216\n"
    )

    summary_lines.append(
        f"Overall: "
        f"{total_correct}/"
        f"{total_trials}\n"
    )

    summary_lines.append(
        f"Accuracy: "
        f"{overall_accuracy * 100:.2f}%\n\n"
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

        finite_confidence = (
            pd.to_numeric(
                subset[
                    "confidence"
                ],
                errors="coerce"
            )
            .dropna()
        )

        if len(
            finite_confidence
        ):

            mean_confidence = float(
                finite_confidence.mean()
            )

            confidence_text = (
                f"{mean_confidence * 100:.1f}%"
            )

        else:

            confidence_text = (
                "N/A"
            )

        line = (
            f"{digit}: "
            f"{correct_count}/"
            f"{len(subset)} "
            f"({digit_accuracy * 100:.1f}%) "
            f"| mean confidence "
            f"{confidence_text}"
        )

        print(
            line
        )

        summary_lines.append(
            line
            + "\n"
        )

    summary_lines.append(
        "\nRaw data: "
        + raw_file
        + "\n"
    )

    summary_lines.append(
        "Results: "
        + results_file
        + "\n"
    )

    Path(
        summary_file
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
