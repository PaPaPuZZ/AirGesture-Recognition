import pandas as pd

RESULTS_FILE = (
    "numbers_6_0_authoritative_certification_results_20261005_164811.csv"
)

TARGET_DIGITS = ["8", "9", "0"]

df = pd.read_csv(RESULTS_FILE)

df["actual"] = (
    df["actual"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

df["prediction"] = (
    df["prediction"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

print()
print("=" * 70)
print("6-0 CERTIFICATION MISCLASSIFICATION DIAGNOSTIC")
print("=" * 70)

print()
print("Rows:", len(df))

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

classes = ["6", "7", "8", "9", "0"]

cm = pd.crosstab(
    df["actual"],
    df["prediction"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
)

cm = cm.reindex(
    index=classes,
    columns=classes,
    fill_value=0
)

print()
print(cm)

print()
print("=" * 70)
print("PER-CLASS PREDICTION BREAKDOWN")
print("=" * 70)

for digit in TARGET_DIGITS:

    subset = df[
        df["actual"] == digit
    ]

    print()
    print(f"Actual digit {digit}:")
    print(
        subset["prediction"]
        .value_counts()
        .sort_index()
    )

print()
print("=" * 70)
print("WRONG 8 / 9 / 0 TRIALS")
print("=" * 70)

wrong = df[
    (df["actual"].isin(TARGET_DIGITS))
    &
    (df["actual"] != df["prediction"])
].copy()

if len(wrong) == 0:

    print()
    print("No wrong 8/9/0 trials found.")

else:

    columns = [
        "actual",
        "trial_number",
        "prediction",
        "confidence",
    ]

    probability_columns = [
        column
        for column in df.columns
        if column.startswith("prob_")
    ]

    columns.extend(
        probability_columns
    )

    print()
    print(
        wrong[columns]
        .to_string(index=False)
    )

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

for digit in TARGET_DIGITS:

    subset = df[
        df["actual"] == digit
    ]

    correct = int(
        (
            subset["actual"]
            == subset["prediction"]
        ).sum()
    )

    total = len(subset)

    wrong_predictions = (
        subset[
            subset["actual"]
            != subset["prediction"]
        ]["prediction"]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        f"{digit}: "
        f"{correct}/{total} correct"
    )

    print(
        "Misclassified as:",
        wrong_predictions
    )

print()
print("Diagnostic complete.")
