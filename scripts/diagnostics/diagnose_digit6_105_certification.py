import pandas as pd

RESULTS_FILE = (
    "numbers_6_0_105_certification_results_20261008_222852.csv"
)

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

digit6 = df[
    df["actual"] == "6"
].copy()

print()
print("=" * 70)
print("DIGIT 6 — 105-MODEL CERTIFICATION DIAGNOSTIC")
print("=" * 70)

print()
print("Digit 6 trials found:", len(digit6))

print()
print("=" * 70)
print("PREDICTION BREAKDOWN")
print("=" * 70)

print(
    digit6["prediction"]
    .value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("TRIAL-BY-TRIAL RESULTS")
print("=" * 70)

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
    digit6[columns]
    .to_string(index=False)
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

correct = int(
    (
        digit6["prediction"] == "6"
    ).sum()
)

print()
print(
    f"Correct: {correct}/{len(digit6)}"
)

wrong_counts = (
    digit6[
        digit6["prediction"] != "6"
    ]["prediction"]
    .value_counts()
    .to_dict()
)

print(
    "Misclassified as:",
    wrong_counts
)

print()
print(
    "Mean confidence:",
    f"{digit6['confidence'].mean() * 100:.2f}%"
)

print()
print("Diagnostic complete.")
