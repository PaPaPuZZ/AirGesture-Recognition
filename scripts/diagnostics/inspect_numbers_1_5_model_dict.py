import pickle
from pathlib import Path

MODEL = "mpu6050_numbers_1_5_best_model.pkl"

print("=" * 70)
print("NUMBERS 1-5 SAVED MODEL DICTIONARY INSPECTION")
print("=" * 70)

with open(MODEL, "rb") as f:
    obj = pickle.load(f)

print("\nTop-level type:", type(obj))

if not isinstance(obj, dict):
    print("Saved object is not a dictionary.")
    raise SystemExit

print("\nDictionary keys:")
for key in obj.keys():
    print(" -", key)

print("\n" + "=" * 70)
print("CONTENTS")
print("=" * 70)

for key, value in obj.items():

    print(f"\nKEY: {key}")
    print("Type:", type(value))

    if hasattr(value, "n_features_in_"):
        print(
            "n_features_in_:",
            value.n_features_in_
        )

    if hasattr(value, "classes_"):
        print(
            "classes_:",
            list(value.classes_)
        )

    if hasattr(value, "steps"):
        print("Pipeline steps:")

        for name, step in value.steps:

            print(
                " ",
                name,
                "->",
                type(step)
            )

            if hasattr(
                step,
                "n_features_in_"
            ):
                print(
                    "    n_features_in_:",
                    step.n_features_in_
                )

            if hasattr(
                step,
                "classes_"
            ):
                print(
                    "    classes_:",
                    list(step.classes_)
                )

            if hasattr(
                step,
                "kernel"
            ):
                print(
                    "    kernel:",
                    step.kernel
                )

            if hasattr(
                step,
                "C"
            ):
                print(
                    "    C:",
                    step.C
                )

            if hasattr(
                step,
                "gamma"
            ):
                print(
                    "    gamma:",
                    step.gamma
                )

    elif isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            list,
            tuple
        )
    ):

        print("Value:", value)

print("\nInspection complete.")
print("Nothing was modified.")
