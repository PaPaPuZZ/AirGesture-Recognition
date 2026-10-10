# ================================================================
# MPU6050 NUMBER 1-5 MODEL CANDIDATE TRAINING + VALIDATION
# ================================================================
#
# PURPOSE:
#   1. Load the 50 collected number recordings: 1,2,3,4,5.
#   2. Validate the raw dataset.
#   3. Convert each recording from 100 x 6 raw MPU6050 values
#      into the same 216 temporal features used by the D-L system.
#   4. Evaluate three candidate classifiers:
#
#         A. Logistic Regression
#         B. Linear SVM
#         C. Random Forest
#
#   5. Use STRATIFIED 5-FOLD CROSS-VALIDATION.
#   6. Generate an out-of-fold prediction for EVERY recording.
#   7. Compare model agreement.
#   8. Identify individual recordings that may warrant re-recording.
#   9. Select the strongest model.
#  10. Train that model on all 50 number recordings.
#  11. Save the final separate number 1-5 model.
#
# IMPORTANT:
#
#   THIS SCRIPT DOES NOT MODIFY THE A-L DATASET.
#   THIS SCRIPT DOES NOT MODIFY THE FROZEN A-L MODEL.
#   THIS SCRIPT DOES NOT COMBINE LETTERS AND NUMBERS.
#
# ================================================================


# ================================================================
# 0. IMPORTS
# ================================================================

import os
import sys
import warnings
import pickle

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ================================================================
# 1. SETTINGS
# ================================================================

INPUT_FILE = "numbers_1_5_50ms_training_raw_data.csv"

FINAL_MODEL_FILE = "mpu6050_numbers_1_5_best_model.pkl"

CV_SUMMARY_FILE = "numbers_1_5_candidate_cv_summary.csv"

OOF_RESULTS_FILE = "numbers_1_5_oof_trial_results.csv"

RERECORD_FILE = "numbers_1_5_rerecord_recommendations.csv"

SUMMARY_FILE = "numbers_1_5_training_summary.txt"


NUM_SAMPLES = 100
NUM_CHANNELS = 6
NUM_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 25

FEATURES_PER_CHANNEL_PER_SEGMENT = 9

EXPECTED_SENSOR_VALUES = (
    NUM_SAMPLES
    * NUM_CHANNELS
)

EXPECTED_FEATURES = (
    NUM_SEGMENTS
    * NUM_CHANNELS
    * FEATURES_PER_CHANNEL_PER_SEGMENT
)

EXPECTED_RECORDINGS = 90

EXPECTED_COUNTS = {
    "1": 10,
    "2": 20,
    "3": 20,
    "4": 20,
    "5": 20,
}

LABELS = [
    "1",
    "2",
    "3",
    "4",
    "5",
]

RANDOM_STATE = 42

N_SPLITS = 5


# ================================================================
# DISPLAY HELPERS
# ================================================================

LINE = "=" * 78
SHORT_LINE = "-" * 78


def section(title):

    print()
    print(LINE)
    print(title)
    print(LINE)


def subsection(title):

    print()
    print(SHORT_LINE)
    print(title)
    print(SHORT_LINE)


# ================================================================
# HEADER
# ================================================================

print()
print(LINE)
print("MPU6050 NUMBER 1-5 THREE-CANDIDATE MODEL TRAINING")
print(LINE)

print(
    """
Purpose:

1. Load the complete 90-recording Number 1-5 dataset.
2. Validate all recordings.
3. Generate the historical 216 temporal features.
4. Evaluate Logistic Regression.
5. Evaluate Linear SVM.
6. Evaluate Random Forest.
7. Produce out-of-fold predictions for every recording.
8. Identify suspicious trials using agreement across models.
9. Select the strongest candidate.
10. Train the winning candidate on all 90 recordings.
11. Save a SEPARATE Number 1-5 model.

IMPORTANT:

The A-L model is NOT changed.
The A-L dataset is NOT changed.
No letter recordings are being retrained here.
"""
)


# ================================================================
# 2. FILE VALIDATION
# ================================================================

section("1. FILE VALIDATION")

if not os.path.exists(INPUT_FILE):

    print()
    print("ERROR:")
    print(f"Could not find:")
    print(f"    {INPUT_FILE}")
    print()

    sys.exit(1)

print(f"PASS: {INPUT_FILE}")


# ================================================================
# 3. LOAD DATA
# ================================================================

section("2. LOADING NUMBER 1-5 DATASET")

df = pd.read_csv(INPUT_FILE)

print()
print(f"Shape: {df.shape}")


# ================================================================
# 4. REQUIRED METADATA
# ================================================================

section("3. METADATA VALIDATION")

if "actual" not in df.columns:

    raise ValueError(
        "The CSV must contain an 'actual' column."
    )


# ------------------------------------------------
# Force number labels into consistent STRING form.
#
# This prevents:
#
#     1 versus "1"
#
# label mismatches inside scikit-learn.
# ------------------------------------------------

df["actual"] = (
    df["actual"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------
# Handle trial number
# ------------------------------------------------

if "trial_number" not in df.columns:

    print(
        "WARNING: No trial_number column found."
    )

    print(
        "Generating trial numbers independently "
        "within each class."
    )

    df["trial_number"] = (
        df.groupby("actual")
        .cumcount()
        + 1
    )

else:

    df["trial_number"] = pd.to_numeric(
        df["trial_number"],
        errors="coerce"
    )

    if df["trial_number"].isna().any():

        raise ValueError(
            "trial_number contains invalid values."
        )

    df["trial_number"] = (
        df["trial_number"]
        .astype(int)
    )


print()
print("Class counts:")

class_counts = (
    df["actual"]
    .value_counts()
    .sort_index()
)

print(class_counts)


# ================================================================
# 5. DATASET VALIDATION
# ================================================================

section("4. DATASET VALIDATION")

if len(df) != EXPECTED_RECORDINGS:

    raise ValueError(
        f"Expected {EXPECTED_RECORDINGS} recordings, "
        f"but found {len(df)}."
    )

print(
    f"PASS: Dataset contains exactly "
    f"{EXPECTED_RECORDINGS} recordings."
)


for label, expected_count in EXPECTED_COUNTS.items():

    observed = int(
        (df["actual"] == label).sum()
    )

    if observed != expected_count:

        raise ValueError(
            f"Class {label}: expected "
            f"{expected_count}, found {observed}."
        )

    print(
        f"PASS: {label} = "
        f"{observed}/{expected_count}"
    )


missing_total = int(
    df.isna().sum().sum()
)

print()
print(
    f"Missing values: {missing_total}"
)

if missing_total != 0:

    raise ValueError(
        "Dataset contains missing values."
    )

print(
    "PASS: No missing values detected."
)


# ================================================================
# 6. IDENTIFY SENSOR COLUMNS
# ================================================================

section("5. IDENTIFYING RAW MPU6050 SENSOR COLUMNS")

metadata_columns = {
    "actual",
    "trial_number",
}


sensor_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print()
print(
    f"Sensor columns detected: "
    f"{len(sensor_columns)}"
)


if len(sensor_columns) != EXPECTED_SENSOR_VALUES:

    raise ValueError(
        f"Expected {EXPECTED_SENSOR_VALUES} sensor "
        f"columns but found {len(sensor_columns)}."
    )


print(
    f"First sensor column: "
    f"{sensor_columns[0]}"
)

print(
    f"Last sensor column: "
    f"{sensor_columns[-1]}"
)


print()
print(
    "IMPORTANT:"
)

print(
    "Sensor columns are preserved in the "
    "original CSV left-to-right order."
)

print(
    "No alphabetical sorting or rearrangement "
    "is performed."
)


# ================================================================
# 7. RAW SENSOR MATRIX
# ================================================================

section("6. RESHAPING SENSOR DATA")

X_raw = (
    df[sensor_columns]
    .apply(
        pd.to_numeric,
        errors="raise"
    )
    .to_numpy(
        dtype=float
    )
)


if not np.isfinite(X_raw).all():

    raise ValueError(
        "Raw sensor matrix contains NaN "
        "or infinite values."
    )


print()
print(
    f"Raw sensor matrix shape: "
    f"{X_raw.shape}"
)


X_sensor = X_raw.reshape(
    len(df),
    NUM_SAMPLES,
    NUM_CHANNELS
)


print(
    f"Sensor array shape: "
    f"{X_sensor.shape}"
)


if X_sensor.shape != (
    EXPECTED_RECORDINGS,
    NUM_SAMPLES,
    NUM_CHANNELS
):

    raise ValueError(
        "Unexpected sensor array shape."
    )


print(
    "PASS: All recordings reshaped to "
    "50 x 100 x 6."
)


# ================================================================
# 8. EXACT TEMPORAL FEATURE EXTRACTION
# ================================================================

section(
    "7. EXTRACTING 216 TEMPORAL FEATURES"
)


def extract_segment_features(segment):

    """
    EXACT 9-feature structure used by the
    historical D-L temporal representation.

    For each of 6 channels:

        1. mean
        2. standard deviation
        3. minimum
        4. maximum
        5. range / peak-to-peak
        6. RMS
        7. average absolute change
        8. maximum absolute change
        9. total movement

    One temporal segment:

        6 channels x 9 features
        = 54 features

    Four temporal segments:

        54 x 4
        = 216 features
    """

    features = []

    for channel_index in range(
        NUM_CHANNELS
    ):

        x = np.asarray(
            segment[:, channel_index],
            dtype=float
        )

        dx = np.diff(x)

        mean_value = np.mean(x)

        std_value = np.std(x)

        min_value = np.min(x)

        max_value = np.max(x)

        range_value = np.ptp(x)

        rms_value = np.sqrt(
            np.mean(
                x ** 2
            )
        )

        if len(dx) > 0:

            avg_abs_change = np.mean(
                np.abs(dx)
            )

            max_abs_change = np.max(
                np.abs(dx)
            )

            total_movement = np.sum(
                np.abs(dx)
            )

        else:

            avg_abs_change = 0.0
            max_abs_change = 0.0
            total_movement = 0.0


        features.extend(
            [
                mean_value,
                std_value,
                min_value,
                max_value,
                range_value,
                rms_value,
                avg_abs_change,
                max_abs_change,
                total_movement,
            ]
        )

    return features



def extract_temporal_features(sensor_array):

    """
    sensor_array:

        recordings x 100 samples x 6 channels

    returns:

        recordings x 216 features
    """

    all_features = []

    for recording in sensor_array:

        recording_features = []

        for segment_index in range(
            NUM_SEGMENTS
        ):

            start_index = (
                segment_index
                * SAMPLES_PER_SEGMENT
            )

            end_index = (
                start_index
                + SAMPLES_PER_SEGMENT
            )

            segment = recording[
                start_index:end_index,
                :
            ]

            segment_features = (
                extract_segment_features(
                    segment
                )
            )

            recording_features.extend(
                segment_features
            )

        all_features.append(
            recording_features
        )


    X = np.asarray(
        all_features,
        dtype=float
    )


    if X.shape[1] != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FEATURES} "
            f"features, found {X.shape[1]}."
        )


    if not np.isfinite(X).all():

        raise ValueError(
            "Feature matrix contains "
            "non-finite values."
        )


    return X



X = extract_temporal_features(
    X_sensor
)

y = (
    df["actual"]
    .to_numpy(
        dtype=str
    )
)


print()
print(
    f"Feature matrix shape: "
    f"{X.shape}"
)

print(
    "PASS: 216 finite temporal features "
    "generated per recording."
)


# ================================================================
# 9. DEFINE THREE MODEL CANDIDATES
# ================================================================

section("8. DEFINING THREE MODEL CANDIDATES")


models = {


    # ------------------------------------------------------------
    # Candidate 1
    # ------------------------------------------------------------

    "Logistic Regression":

        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",

                    LogisticRegression(
                        C=1.0,
                        max_iter=10000,
                        random_state=RANDOM_STATE
                    )
                ),
            ]
        ),


    # ------------------------------------------------------------
    # Candidate 2
    # ------------------------------------------------------------

    "Linear SVM":

        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",

                    SVC(
                        kernel="linear",
                        C=1.0,
                        probability=True,
                        random_state=RANDOM_STATE
                    )
                ),
            ]
        ),


    # ------------------------------------------------------------
    # Candidate 3
    #
    # IMPORTANT:
    #
    # class_weight is intentionally omitted.
    #
    # Classes are already perfectly balanced:
    #
    #     1 = 10
    #     2 = 10
    #     3 = 10
    #     4 = 10
    #     5 = 10
    #
    # ------------------------------------------------------------

    "Random Forest":

        RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
}


for name in models:

    print(
        f"PASS: {name}"
    )


# ================================================================
# 10. STRATIFIED CROSS-VALIDATION
# ================================================================

section(
    "9. STRATIFIED FIVE-FOLD CROSS-VALIDATION"
)


cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


summary_rows = []

all_model_oof = {}


for model_name, base_model in models.items():

    subsection(model_name)

    oof_prediction = np.empty(
        len(y),
        dtype=object
    )

    oof_confidence = np.full(
        len(y),
        np.nan,
        dtype=float
    )

    oof_correct_probability = np.full(
        len(y),
        np.nan,
        dtype=float
    )

    fold_accuracies = []

    fold_balanced_accuracies = []


    for fold_number, (
        train_indices,
        test_indices
    ) in enumerate(
        cv.split(
            X,
            y
        ),
        start=1
    ):

        model = clone(
            base_model
        )

        X_train = X[
            train_indices
        ]

        X_test = X[
            test_indices
        ]

        y_train = y[
            train_indices
        ]

        y_test = y[
            test_indices
        ]


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        probabilities = (
            model.predict_proba(
                X_test
            )
        )


        model_classes = [
            str(x)
            for x
            in model.classes_
        ]


        confidence = np.max(
            probabilities,
            axis=1
        )


        correct_probability = []

        for row_index, true_label in enumerate(
            y_test
        ):

            class_index = (
                model_classes.index(
                    str(true_label)
                )
            )

            correct_probability.append(
                probabilities[
                    row_index,
                    class_index
                ]
            )


        correct_probability = np.asarray(
            correct_probability,
            dtype=float
        )


        oof_prediction[
            test_indices
        ] = predictions

        oof_confidence[
            test_indices
        ] = confidence

        oof_correct_probability[
            test_indices
        ] = correct_probability


        fold_accuracy = accuracy_score(
            y_test,
            predictions
        )

        fold_balanced = (
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )


        fold_accuracies.append(
            fold_accuracy
        )

        fold_balanced_accuracies.append(
            fold_balanced
        )


        print(
            f"Fold {fold_number}: "
            f"{int(np.sum(predictions == y_test))}"
            f"/{len(y_test)} "
            f"({100 * fold_accuracy:.2f}%)"
        )


    overall_accuracy = accuracy_score(
        y,
        oof_prediction
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            oof_prediction
        )
    )


    print()
    print(
        f"OOF accuracy: "
        f"{int(np.sum(oof_prediction == y))}"
        f"/{len(y)} "
        f"({100 * overall_accuracy:.2f}%)"
    )

    print(
        f"Balanced accuracy: "
        f"{100 * balanced_accuracy:.2f}%"
    )

    print(
        f"Mean fold accuracy: "
        f"{100 * np.mean(fold_accuracies):.2f}%"
    )

    print(
        f"Fold accuracy SD: "
        f"{100 * np.std(fold_accuracies):.2f}%"
    )


    summary_rows.append(
        {
            "model": model_name,

            "oof_accuracy":
                overall_accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "mean_fold_accuracy":
                np.mean(
                    fold_accuracies
                ),

            "std_fold_accuracy":
                np.std(
                    fold_accuracies
                ),

            "mean_prediction_confidence":
                np.mean(
                    oof_confidence
                ),

            "mean_correct_class_probability":
                np.mean(
                    oof_correct_probability
                ),
        }
    )


    all_model_oof[
        model_name
    ] = {

        "prediction":
            np.asarray(
                oof_prediction,
                dtype=str
            ),

        "confidence":
            oof_confidence,

        "correct_probability":
            oof_correct_probability,
    }


# ================================================================
# 11. CV SUMMARY
# ================================================================

section(
    "10. CROSS-VALIDATION SUMMARY"
)


summary_df = pd.DataFrame(
    summary_rows
)


summary_df = (
    summary_df
    .sort_values(
        [
            "oof_accuracy",
            "balanced_accuracy",
            "mean_correct_class_probability",
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print()
print(
    summary_df.to_string(
        index=False,
        formatters={
            "oof_accuracy":
                lambda x:
                    f"{100*x:.2f}%",

            "balanced_accuracy":
                lambda x:
                    f"{100*x:.2f}%",

            "mean_fold_accuracy":
                lambda x:
                    f"{100*x:.2f}%",

            "std_fold_accuracy":
                lambda x:
                    f"{100*x:.2f}%",

            "mean_prediction_confidence":
                lambda x:
                    f"{100*x:.2f}%",

            "mean_correct_class_probability":
                lambda x:
                    f"{100*x:.2f}%",
        }
    )
)


summary_df.to_csv(
    CV_SUMMARY_FILE,
    index=False
)


print()
print(
    f"Saved: {CV_SUMMARY_FILE}"
)


# ================================================================
# 12. BEST MODEL
# ================================================================

section(
    "11. MODEL SELECTION"
)


best_model_name = (
    summary_df.iloc[0][
        "model"
    ]
)


best_accuracy = float(
    summary_df.iloc[0][
        "oof_accuracy"
    ]
)


best_balanced_accuracy = float(
    summary_df.iloc[0][
        "balanced_accuracy"
    ]
)


print()
print(
    f"Selected candidate: "
    f"{best_model_name}"
)

print(
    f"OOF accuracy: "
    f"{100 * best_accuracy:.2f}%"
)

print(
    f"Balanced accuracy: "
    f"{100 * best_balanced_accuracy:.2f}%"
)


# ================================================================
# 13. CONFUSION MATRICES
# ================================================================

section(
    "12. OUT-OF-FOLD CONFUSION MATRICES"
)


for model_name in models:

    subsection(
        model_name
    )

    predictions = (
        all_model_oof[
            model_name
        ][
            "prediction"
        ]
    )


    matrix = confusion_matrix(
        y,
        predictions,
        labels=LABELS
    )


    print()
    print(matrix)

    print()
    print(
        "Rows    = actual 1 2 3 4 5"
    )

    print(
        "Columns = predicted 1 2 3 4 5"
    )


# ================================================================
# 14. CLASSIFICATION REPORTS
# ================================================================

section(
    "13. OUT-OF-FOLD CLASSIFICATION REPORTS"
)


for model_name in models:

    subsection(
        model_name
    )

    predictions = (
        all_model_oof[
            model_name
        ][
            "prediction"
        ]
    )


    report = classification_report(
        y,
        predictions,
        labels=LABELS,
        zero_division=0,
        digits=3
    )

    print(
        report
    )


# ================================================================
# 15. BUILD RECORDING-BY-RECORDING TABLE
# ================================================================

section(
    "14. INDIVIDUAL RECORDING DIAGNOSTICS"
)


result_df = pd.DataFrame(
    {
        "actual":
            df["actual"]
            .astype(str)
            .values,

        "trial_number":
            df["trial_number"]
            .astype(int)
            .values,
    }
)


model_short_names = {

    "Logistic Regression":
        "logistic",

    "Linear SVM":
        "svm",

    "Random Forest":
        "random_forest",
}


for model_name, short_name in (
    model_short_names.items()
):

    predictions = (
        all_model_oof[
            model_name
        ][
            "prediction"
        ]
    )

    confidences = (
        all_model_oof[
            model_name
        ][
            "confidence"
        ]
    )

    correct_probabilities = (
        all_model_oof[
            model_name
        ][
            "correct_probability"
        ]
    )


    result_df[
        f"{short_name}_prediction"
    ] = predictions


    result_df[
        f"{short_name}_correct"
    ] = (
        predictions
        == y
    )


    result_df[
        f"{short_name}_confidence"
    ] = confidences


    result_df[
        f"{short_name}_correct_probability"
    ] = correct_probabilities


# ================================================================
# 16. CONSENSUS ERROR ANALYSIS
# ================================================================

section(
    "15. THREE-MODEL CONSENSUS ANALYSIS"
)


correct_columns = [

    "logistic_correct",

    "svm_correct",

    "random_forest_correct",
]


correct_probability_columns = [

    "logistic_correct_probability",

    "svm_correct_probability",

    "random_forest_correct_probability",
]


result_df[
    "models_correct"
] = (
    result_df[
        correct_columns
    ]
    .sum(
        axis=1
    )
)


result_df[
    "models_wrong"
] = (
    3
    - result_df[
        "models_correct"
    ]
)


result_df[
    "mean_correct_class_probability"
] = (
    result_df[
        correct_probability_columns
    ]
    .mean(
        axis=1
    )
)


result_df[
    "all_three_wrong"
] = (
    result_df[
        "models_wrong"
    ]
    == 3
)


result_df[
    "two_or_more_wrong"
] = (
    result_df[
        "models_wrong"
    ]
    >= 2
)


print()
print(
    "Recordings misclassified by all "
    "three candidates:"
)

all_wrong = result_df[
    result_df[
        "all_three_wrong"
    ]
]


if len(all_wrong) == 0:

    print(
        "None."
    )

else:

    print(
        all_wrong[
            [
                "actual",
                "trial_number",
                "logistic_prediction",
                "svm_prediction",
                "random_forest_prediction",
                "mean_correct_class_probability",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ================================================================
# 17. RE-RECORDING DECISION RULE
# ================================================================

section(
    "16. AUTOMATIC RE-RECORDING RECOMMENDATIONS"
)


# ------------------------------------------------
# IMPORTANT:
#
# We do NOT automatically throw away every
# misclassified sample.
#
# A single model may simply have a poor decision
# boundary.
#
# Re-recording priority is based on agreement
# across all three independent classifiers.
#
#
# HIGH PRIORITY
#
#     all 3 models wrong
#
#     AND
#
#     average probability assigned to correct
#     class < 20%
#
#
# MEDIUM PRIORITY
#
#     at least 2 models wrong
#
#     AND
#
#     average correct-class probability < 35%
#
#
# REVIEW
#
#     at least 2 models wrong
#
#     but not sufficiently extreme to justify
#     automatic re-recording.
#
#
# KEEP
#
#     at least 2 models correct
#
# ------------------------------------------------


def determine_recommendation(row):

    wrong = int(
        row[
            "models_wrong"
        ]
    )

    probability = float(
        row[
            "mean_correct_class_probability"
        ]
    )


    if (
        wrong == 3
        and probability < 0.20
    ):

        return "HIGH PRIORITY RE-RECORD"


    elif (
        wrong >= 2
        and probability < 0.35
    ):

        return "MEDIUM PRIORITY RE-RECORD"


    elif wrong >= 2:

        return "REVIEW"


    else:

        return "KEEP"



result_df[
    "recommendation"
] = result_df.apply(
    determine_recommendation,
    axis=1
)


priority_order = {

    "HIGH PRIORITY RE-RECORD":
        0,

    "MEDIUM PRIORITY RE-RECORD":
        1,

    "REVIEW":
        2,

    "KEEP":
        3,
}


result_df[
    "recommendation_rank"
] = (
    result_df[
        "recommendation"
    ]
    .map(
        priority_order
    )
)


display_df = (
    result_df
    .sort_values(
        [
            "recommendation_rank",
            "actual",
            "trial_number",
        ]
    )
)


print()

print(
    display_df[
        [
            "actual",
            "trial_number",
            "logistic_prediction",
            "svm_prediction",
            "random_forest_prediction",
            "models_wrong",
            "mean_correct_class_probability",
            "recommendation",
        ]
    ]
    .to_string(
        index=False
    )
)


# ================================================================
# 18. EXACT TRIALS TO RE-RECORD
# ================================================================

section(
    "17. EXACT TRIALS RECOMMENDED FOR RE-RECORDING"
)


rerecord_df = result_df[

    result_df[
        "recommendation"
    ].isin(
        [
            "HIGH PRIORITY RE-RECORD",
            "MEDIUM PRIORITY RE-RECORD",
        ]
    )

].copy()


rerecord_df = (
    rerecord_df
    .sort_values(
        [
            "actual",
            "trial_number",
        ]
    )
)


if len(rerecord_df) == 0:

    print()
    print(
        "NO RECORDINGS ARE CURRENTLY "
        "RECOMMENDED FOR RE-RECORDING."
    )

    print()
    print(
        "The next step should be live "
        "Number 1-5 testing."
    )

else:

    print()
    print(
        f"{len(rerecord_df)} recording(s) "
        f"recommended for replacement:"
    )

    print()

    for _, row in (
        rerecord_df.iterrows()
    ):

        print(
            f"NUMBER {row['actual']} "
            f"TRIAL {int(row['trial_number'])}"
            f"  ->  "
            f"{row['recommendation']}"
        )


    print()
    print(
        "Only replace the trials listed above."
    )

    print(
        "Do NOT recollect the entire dataset."
    )


# ================================================================
# 19. CLASS-BY-CLASS ERROR ANALYSIS
# ================================================================

section(
    "18. CLASS-BY-CLASS CONSENSUS SUMMARY"
)


for label in LABELS:

    subset = result_df[
        result_df[
            "actual"
        ]
        == label
    ]

    print()
    print(
        f"NUMBER {label}"
    )

    print(
        SHORT_LINE
    )

    print(
        f"Recordings: "
        f"{len(subset)}"
    )

    print(
        f"All 3 models wrong: "
        f"{int(subset['all_three_wrong'].sum())}"
    )

    print(
        f"2+ models wrong: "
        f"{int(subset['two_or_more_wrong'].sum())}"
    )

    print(
        f"Mean correct-class probability: "
        f"{100 * subset['mean_correct_class_probability'].mean():.2f}%"
    )

    print(
        f"Recommended re-records: "
        f"{int(subset['recommendation'].isin(
            [
                'HIGH PRIORITY RE-RECORD',
                'MEDIUM PRIORITY RE-RECORD'
            ]
        ).sum())}"
    )


# ================================================================
# 20. TRAIN FINAL WINNING MODEL
# ================================================================

section(
    "19. TRAINING FINAL NUMBER 1-5 MODEL"
)


print()
print(
    "Training ONLY the selected Number 1-5 "
    "candidate on all 90 number recordings."
)

print(
    "No A-L data or model is involved."
)


final_model = clone(
    models[
        best_model_name
    ]
)


final_model.fit(
    X,
    y
)


print()
print(
    f"PASS: {best_model_name} fitted "
    f"on all {len(y)} Number 1-5 recordings."
)


# ================================================================
# 21. FULL-DATA SANITY CHECK
# ================================================================

section(
    "20. TRAINING-SET SANITY CHECK"
)


training_predictions = (
    final_model.predict(
        X
    )
)


training_accuracy = (
    accuracy_score(
        y,
        training_predictions
    )
)


print()
print(
    f"Training-set accuracy: "
    f"{int(np.sum(training_predictions == y))}"
    f"/{len(y)} "
    f"({100 * training_accuracy:.2f}%)"
)


print()
print(
    "NOTE:"
)

print(
    "This is NOT the validation score."
)

print(
    "The cross-validation results above "
    "remain the meaningful performance estimate."
)


# ================================================================
# 22. SAVE FINAL MODEL PACKAGE
# ================================================================

section(
    "21. SAVING FINAL NUMBER MODEL"
)


model_package = {

    "model":
        final_model,

    "model_name":
        best_model_name,

    "classes":
        LABELS,

    "num_samples":
        NUM_SAMPLES,

    "num_channels":
        NUM_CHANNELS,

    "num_segments":
        NUM_SEGMENTS,

    "samples_per_segment":
        SAMPLES_PER_SEGMENT,

    "features_per_channel_per_segment":
        FEATURES_PER_CHANNEL_PER_SEGMENT,

    "expected_features":
        EXPECTED_FEATURES,

    "sensor_columns":
        sensor_columns,

    "cv_oof_accuracy":
        best_accuracy,

    "cv_balanced_accuracy":
        best_balanced_accuracy,

    "feature_definition":
        [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "rms",
            "avg_abs_change",
            "max_abs_change",
            "total_movement",
        ],
}


with open(
    FINAL_MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model_package,
        file
    )


print()
print(
    f"Saved model: "
    f"{FINAL_MODEL_FILE}"
)


# ================================================================
# 23. SAVE DETAILED TABLES
# ================================================================

section(
    "22. SAVING DIAGNOSTIC TABLES"
)


result_df.drop(
    columns=[
        "recommendation_rank"
    ]
).to_csv(
    OOF_RESULTS_FILE,
    index=False
)


rerecord_df.drop(
    columns=[
        "recommendation_rank"
    ],
    errors="ignore"
).to_csv(
    RERECORD_FILE,
    index=False
)


print()
print(
    f"Saved OOF trial results:"
)

print(
    f"    {OOF_RESULTS_FILE}"
)

print()
print(
    f"Saved re-record recommendations:"
)

print(
    f"    {RERECORD_FILE}"
)


# ================================================================
# 24. FINAL DECISION
# ================================================================

section(
    "23. FINAL DECISION"
)


print()
print(
    f"BEST CANDIDATE:"
)

print(
    f"    {best_model_name}"
)


print()

print(
    f"CROSS-VALIDATED ACCURACY:"
)

print(
    f"    {100 * best_accuracy:.2f}%"
)


print()

print(
    f"BALANCED ACCURACY:"
)

print(
    f"    {100 * best_balanced_accuracy:.2f}%"
)


print()

print(
    f"TRIALS RECOMMENDED FOR RE-RECORDING:"
)

print(
    f"    {len(rerecord_df)}"
)


if len(rerecord_df) > 0:

    print()

    for _, row in (
        rerecord_df.iterrows()
    ):

        print(
            f"    Number {row['actual']} "
            f"Trial {int(row['trial_number'])}"
        )


else:

    print()
    print(
        "    None."
    )


# ================================================================
# PERFORMANCE INTERPRETATION
# ================================================================

print()
print(
    SHORT_LINE
)


if best_accuracy >= 0.90:

    assessment = "EXCELLENT"

    next_step = (
        "Proceed directly to live Number 1-5 "
        "prediction testing."
    )


elif best_accuracy >= 0.80:

    assessment = "GOOD"

    next_step = (
        "Replace only any trials specifically "
        "flagged above, then proceed to live testing."
    )


elif best_accuracy >= 0.70:

    assessment = "PROMISING"

    next_step = (
        "Replace the flagged trials and rerun "
        "this validation before live deployment."
    )


elif best_accuracy >= 0.60:

    assessment = "MODERATE"

    next_step = (
        "Target only the classes/trials with "
        "consistent cross-model errors."
    )


else:

    assessment = "WEAK"

    next_step = (
        "Do not blindly recollect all recordings. "
        "Inspect the cross-model error pattern first."
    )


print()

print(
    f"Assessment: {assessment}"
)

print()

print(
    f"Recommended next step:"
)

print(
    f"{next_step}"
)


# ================================================================
# 25. SAVE TEXT SUMMARY
# ================================================================

summary_lines = []


summary_lines.append(
    "MPU6050 NUMBER 1-5 MODEL TRAINING SUMMARY"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Input dataset: {INPUT_FILE}"
)

summary_lines.append(
    f"Recordings: {len(df)}"
)

summary_lines.append(
    f"Features per recording: {EXPECTED_FEATURES}"
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Best candidate: {best_model_name}"
)

summary_lines.append(
    f"OOF accuracy: {100 * best_accuracy:.2f}%"
)

summary_lines.append(
    f"Balanced accuracy: "
    f"{100 * best_balanced_accuracy:.2f}%"
)

summary_lines.append(
    f"Assessment: {assessment}"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Re-record recommendations:"
)


if len(rerecord_df) == 0:

    summary_lines.append(
        "None."
    )

else:

    for _, row in (
        rerecord_df.iterrows()
    ):

        summary_lines.append(
            f"Number {row['actual']} "
            f"Trial {int(row['trial_number'])} "
            f"- {row['recommendation']}"
        )


summary_lines.append(
    ""
)

summary_lines.append(
    f"Final model saved: {FINAL_MODEL_FILE}"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "IMPORTANT:"
)

summary_lines.append(
    "The A-L model was not modified."
)

summary_lines.append(
    "The A-L training dataset was not modified."
)

summary_lines.append(
    "The number model remains separate."
)


with open(
    SUMMARY_FILE,
    "w"
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


print()
print(
    f"Saved summary:"
)

print(
    f"    {SUMMARY_FILE}"
)


# ================================================================
# COMPLETE
# ================================================================

print()
print(LINE)
print(
    "NUMBER 1-5 THREE-CANDIDATE TRAINING COMPLETE"
)
print(LINE)

print()

print(
    f"Winning model: "
    f"{best_model_name}"
)

print(
    f"Cross-validated accuracy: "
    f"{100 * best_accuracy:.2f}%"
)

print(
    f"Re-recordings recommended: "
    f"{len(rerecord_df)}"
)

print()

print(
    "Files saved:"
)

print(
    f"1. {FINAL_MODEL_FILE}"
)

print(
    f"2. {CV_SUMMARY_FILE}"
)

print(
    f"3. {OOF_RESULTS_FILE}"
)

print(
    f"4. {RERECORD_FILE}"
)

print(
    f"5. {SUMMARY_FILE}"
)

print()

print(
    "The A-L model remains completely unchanged."
)

print(LINE)
