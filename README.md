# AirGesture Recognition

Senior Design Project for IMU-based air-gesture recognition using MPU6050 sensor data, temporal feature extraction, machine learning, and a wearable glove/PCB system.

## Project Goal

Build a wearable sensing platform that can recognize air-drawn letters, digits, and hand gestures using inertial sensing. The current prototype uses an Arduino Mega 2560 and an MPU6050 IMU, with future integration of a second IMU, BMP280 pressure sensing, microfluidic finger sensors, and a custom PCB.

## Current Signal Pipeline

```text
MPU6050
  ↓
Arduino Mega 2560
  ↓
100 samples × 6 channels
  ↓
50 ms/sample (~5 s gesture window)
  ↓
4 temporal segments
  ↓
9 features/channel/segment
  ↓
216-feature vector
  ↓
Random Forest / Logistic Regression / RBF SVM comparison
  ↓
Gesture prediction
```

The six IMU channels are 3-axis acceleration and 3-axis angular velocity. Current Arduino output uses SI units: acceleration in m/s² and gyroscope values in rad/s.

## Quick Start

Create the software environment with either:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate airgesture
```

Reference Arduino firmware:

- [arduino/mpu6050_capture.ino](arduino/mpu6050_capture.ino)

Current unified numeric workflow:

- [Training script](scripts/training/train_numbers_0_9_candidates.py)
- [Live certification script](scripts/certification/certify_numbers_0_9_unified.py)
- [Unified 0–9 training dataset](data/numbers/numbers_0_9_combined_training_raw_data.csv)
- [Unified 0–9 saved model](models/mpu6050_numbers_0_9_best_model.pkl)
- [Model comparison results](results/numbers/numbers_0_9_model_comparison.csv)
- [OOF confusion matrix](results/numbers/numbers_0_9_oof_confusion_matrix.csv)
- [OOF predictions](results/numbers/numbers_0_9_oof_predictions.csv)

## Temporal Feature Extraction

Every 100×6 recording is split into four 25-sample temporal segments. For each of the six channels in each segment, nine features are calculated:

1. Mean
2. Standard deviation
3. Minimum
4. Maximum
5. Range / peak-to-peak
6. RMS
7. Mean absolute first difference
8. Maximum absolute first difference
9. Sum of absolute first differences / total movement

This produces:

```text
4 segments × 6 channels × 9 features = 216 features
```

See [docs/feature_extraction.md](docs/feature_extraction.md) for the full definition and the feature-consistency lesson from earlier debugging.

## Numeric Recognition Progress

A unified 0–9 training dataset has been assembled with **215 recordings**.

| Digit | Recordings |
|---|---:|
| 0 | 30 |
| 1 | 10 |
| 2 | 20 |
| 3 | 20 |
| 4 | 20 |
| 5 | 20 |
| 6 | 35 |
| 7 | 15 |
| 8 | 20 |
| 9 | 25 |

### Unified 0–9 Model Comparison

The current selected model is **Logistic Regression**.

| Model | OOF Accuracy | OOF Balanced Accuracy | OOF Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 90.70% | 90.91% | 91.28% |
| RBF SVM | 89.77% | 89.51% | 90.13% |
| Random Forest | 88.37% | 88.06% | 89.06% |

A fresh unified 0–9 live certification is the current numeric-model task.

## Letter Recognition Progress

Work completed so far includes A–L collection/training/testing, including investigation into live-vs-training unit and feature-extraction mismatch. The next alphabet milestone is **M–Z data collection and training**.

Useful letter artifacts:

- [D–L 140-recording dataset](data/letters/dl_retraining_140_combined_raw_data.csv)
- [D–L training script](scripts/training/train_dl_retraining_140_temporal.py)
- [D–L saved model](models/dl_retrained_140_temporal_best_model.pkl)
- [Letter testing results](results/letters/)
- [Letter data](data/letters/)

## Earlier ML / Anaconda Work

Earlier exploration included:

- Kaggle air-writing dataset experiments
- Neuton TinyML workflow
- LSTM experiments in Anaconda/Jupyter
- Random Forest comparisons
- Confusion matrices and train/test analysis
- Migration toward the current 216-feature classical ML pipeline

The original notebooks are not yet in the repository. See [notebooks/README.md](notebooks/README.md) for the planned notebook archive.

## Repository Structure

```text
arduino/                  Arduino capture firmware
data/
  letters/                Letter datasets
  numbers/                Numeric datasets and remediation sets
docs/                     Project documentation
models/                   Saved model checkpoints
notebooks/                Anaconda/Jupyter notebook archive
results/
  letters/                Letter model outputs
  numbers/                Numeric model outputs
scripts/
  certification/          Fresh live holdout testing
  data_collection/        Arduino/MPU6050 data collection
  diagnostics/            Debugging and misclassification analysis
  training/               Training and cross-validation
  utilities/              Dataset merge/validation scripts
```

See [scripts/README.md](scripts/README.md) for the current reference workflow and historical-script guidance.

## Documentation

- [Project overview](docs/project_overview.md)
- [Hardware setup](docs/hardware_setup.md)
- [Temporal feature extraction](docs/feature_extraction.md)
- [Model history](docs/model_history.md)
- [Testing results](docs/testing_results.md)

## Current Next Steps

- Complete unified 0–9 live certification
- Return to letters M–Z
- Integrate/test a second MPU6050
- Continue BMP280 and microfluidic sensor integration
- Develop dual-IMU PCB
- Integrate sensors into wearable glove prototype
- Perform full-system validation
