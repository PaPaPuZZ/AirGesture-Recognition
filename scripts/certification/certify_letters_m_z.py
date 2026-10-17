#!/usr/bin/env python3
"""
FRESH LIVE CERTIFICATION FOR LETTERS M-Z

Model:
    models/mpu6050_letters_M_Z_best_model.pkl

Fresh holdout:
    3 trials per letter
    14 letters
    42 total trials

Raw input:
    100 samples x 6 channels

Feature extraction:
    4 temporal segments
    x 6 channels
    x 9 features
    = 216 total features

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

This script does NOT retrain or modify the saved model.
Fresh holdout recordings are saved separately under results/letters/.
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
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    REPO_ROOT
    / "models/mpu6050_letters_M_Z_best_model.pkl"
)

RESULTS_DIR = (
    REPO_ROOT
    / "results/letters"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BAUD_RATE = 115200

NUM_SAMPLES = 100
NUM_CHANNELS = 6

NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

EXPECTED_FEATURES = 216

LETTERS = list(
    "MNOPQRSTUVWXYZ"
)

TRIALS_PER_LETTER = 3

COUNTDOWN_SECONDS = 3

START_TIMEOUT_SECONDS = 8
RECORDING_TIMEOUT_SECONDS = 12


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

    for segment in np.array_split(
        recording,
        NUM_SEGMENTS,
        axis=0
    ):

        for channel in range(
            NUM_CHANNELS
        ):

            x = segment[:, channel]
            dx = np.diff(x)

            features.extend([
                np.mean(x),
                np.std(x),
                np.min(x),
                np.max(x),
                np.ptp(x),
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
            ])

    features = np.asarray(
        features,
        dtype=float
    )

    if features.shape != (
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            f"Expected 216 features, "
            f"got {features.shape}"
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
# MODEL LOADING / VERIFICATION
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Could not find model:\n"
            f"{MODEL_FILE}"
        )

    with open(
        MODEL_FILE,
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
        saved.get(
            "expected_features",
            -1
        )
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "Saved model does not "
            "expect 216 features."
        )

    saved_classes = [
        str(x)
        for x in saved.get(
            "classes",
            []
        )
    ]

    if saved_classes != LETTERS:

        raise RuntimeError(
            f"Expected classes {LETTERS}, "
            f"found {saved_classes}"
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

    return saved, model


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
        set(ports)
    )


def choose_serial_port():

    ports = find_serial_ports()

    if not ports:

        print()
        print(
            "No Arduino serial port found."
        )

        print(
            "Detected /dev/cu.* devices:"
        )

        for port in sorted(
            glob.glob(
                "/dev/cu.*"
            )
        ):

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

    return recording, elapsed


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

    return {
        "mean":
            float(
                np.mean(magnitude)
            ),

        "median":
            float(
                np.median(magnitude)
            ),

        "min":
            float(
                np.min(magnitude)
            ),

        "max":
            float(
                np.max(magnitude)
            ),
    }


# ============================================================
# RAW ROW
# ============================================================

def make_raw_row(
    letter,
    trial,
    recording,
    elapsed,
):

    row = {
        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "actual":
            letter,

        "trial_number":
            trial,

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "LETTERS M-Z LIVE CERTIFICATION"
    )
    print("=" * 72)

    print()
    print(
        f"{TRIALS_PER_LETTER} "
        "fresh trials per letter"
    )

    print(
        f"{len(LETTERS) * TRIALS_PER_LETTER} "
        "total trials"
    )

    saved, model = load_model()

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

    ser = open_serial(
        port
    )

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
        RESULTS_DIR
        / (
            "letters_M_Z_certification_raw_"
            + timestamp
            + ".csv"
        )
    )

    results_file = (
        RESULTS_DIR
        / (
            "letters_M_Z_certification_results_"
            + timestamp
            + ".csv"
        )
    )

    summary_file = (
        RESULTS_DIR
        / (
            "letters_M_Z_certification_summary_"
            + timestamp
            + ".txt"
        )
    )

    raw_rows = []
    result_rows = []

    total_correct = 0

    try:

        for letter in LETTERS:

            print()
            print("=" * 72)
            print(
                f"LETTER {letter}"
            )
            print("=" * 72)

            for trial in range(
                1,
                TRIALS_PER_LETTER + 1
            ):

                while True:

                    print()
                    print(
                        f"Letter {letter} — "
                        f"Trial "
                        f"{trial}/"
                        f"{TRIALS_PER_LETTER}"
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

                    probabilities = None
                    probability_map = {}

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
                            str(label):
                                float(probability)
                            for label, probability
                            in zip(
                                model.classes_,
                                probabilities
                            )
                        }

                    correct = (
                        prediction
                        == letter
                    )

                    if correct:

                        total_correct += 1

                    if probability_map:

                        confidence = (
                            probability_map[
                                prediction
                            ]
                        )

                    else:

                        confidence = np.nan

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

                    if probability_map:

                        ranked = sorted(
                            probability_map.items(),
                            key=lambda item:
                                item[1],
                            reverse=True
                        )

                        print(
                            "Top probabilities:"
                        )

                        for (
                            label,
                            probability
                        ) in ranked[:3]:

                            print(
                                f"  {label}: "
                                f"{probability * 100:.2f}%"
                            )

                    raw_rows.append(
                        make_raw_row(
                            letter,
                            trial,
                            recording,
                            elapsed,
                        )
                    )

                    result = {
                        "actual":
                            letter,

                        "trial_number":
                            trial,

                        "prediction":
                            prediction,

                        "correct":
                            correct,

                        "confidence":
                            confidence,

                        "accel_magnitude_mean":
                            summary["mean"],

                        "accel_magnitude_median":
                            summary["median"],

                        "feature_count":
                            features.size,
                    }

                    for label in model_classes:

                        result[
                            f"prob_{label}"
                        ] = probability_map.get(
                            label,
                            np.nan
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

        try:

            ser.close()

        except Exception:

            pass

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
        "Per-letter results:"
    )

    summary_lines = [
        "LETTERS M-Z LIVE CERTIFICATION\n",
        "=" * 60 + "\n",
        f"Model: {saved.get('model_name')}\n",
        f"Training rows: {saved.get('training_rows')}\n",
        f"Overall: {total_correct}/{total_trials}\n",
        f"Accuracy: {overall_accuracy * 100:.2f}%\n\n",
    ]

    for letter in LETTERS:

        subset = results_df[
            results_df[
                "actual"
            ].astype(str)
            == letter
        ]

        if len(subset) == 0:

            continue

        correct_count = int(
            subset[
                "correct"
            ].sum()
        )

        letter_accuracy = (
            correct_count
            / len(subset)
        )

        mean_confidence = float(
            subset[
                "confidence"
            ].dropna()
            .mean()
        ) if (
            subset[
                "confidence"
            ].notna()
            .any()
        ) else np.nan

        line = (
            f"{letter}: "
            f"{correct_count}/"
            f"{len(subset)} "
            f"({letter_accuracy * 100:.1f}%)"
        )

        if np.isfinite(
            mean_confidence
        ):

            line += (
                f" | mean confidence "
                f"{mean_confidence * 100:.1f}%"
            )

        print(
            line
        )

        summary_lines.append(
            line
            + "\n"
        )

    summary_lines.extend([
        "\nRaw data: "
        + str(raw_file)
        + "\n",

        "Results: "
        + str(results_file)
        + "\n",
    ])

    summary_file.write_text(
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
