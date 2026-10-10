# Model Development History

## Early Anaconda / TinyML Work

The original notebook is preserved at [../notebooks/BuildTrainLSTM.ipynb](../notebooks/BuildTrainLSTM.ipynb).

The notebook includes:
- a 60/40 stratified train/test split
- LabelEncoder-based target encoding
- StandardScaler preprocessing
- reshaping into 100 time steps × 6 IMU channels
- a stacked LSTM with 64-unit and 32-unit recurrent layers
- dropout, batch normalization, and a dense softmax classifier
- test-set accuracy evaluation
- classification report generation
- normalized confusion matrix visualization
- saving the trained Keras model, scaler, and label encoder
- later loading those saved artifacts for prediction on recorded gestures

Earlier work also included:
- Kaggle air-writing dataset exploration
- Neuton TinyML experiments
- Random Forest experiments
- confusion matrices
- train/test split evaluation

## Project-Specific Pipeline

The project later moved to its own 100 × 6 Arduino/MPU6050 recordings and the 216-feature temporal representation.

## D–L

D–L work compared Logistic Regression, Linear SVM, k-NN, RBF SVM, Random Forest, and LDA. High cross-validation accuracy did not initially transfer to live testing because of unit mismatch and later a feature-extractor mismatch. This led to a stricter rule: training, holdout, and live prediction must share one exact extractor.

## Numbers 1–5

The authoritative 1–5 saved model used Random Forest with the 216-feature extractor. Fresh corrected certification reached **21/25 = 84%**.

## Numbers 6–0

The initial 50-recording model selected RBF SVM but live testing exposed weak 8/9/0 generalization.

Targeted remediation added 55 recordings, then another 20 digit-6 recordings after six collapsed into 0/9.

The final 125-recording 6–0 model selected Logistic Regression. Fresh holdout reached **20/25 = 80%**.

## Unified 0–9

The final numeric training set contains 215 recordings across all ten digits.

Selected unified model: **Logistic Regression**

- OOF accuracy: 90.70%
- OOF balanced accuracy: 90.91%
- OOF macro F1: 91.28%

A fresh unified live certification is the next numeric milestone.


## Live Prediction Notebook

The notebook [../notebooks/live_predict.ipynb](../notebooks/live_predict.ipynb) preserves the earlier live A/B/C diagnostic workflow.

It includes:
- loading `live_test_raw_data.csv` with 30 live recordings and 600 raw sensor values per trial
- loading the saved A/B/C Random Forest model and metadata
- reproducing saved predictions and probabilities
- isolating A trials misclassified as C
- comparing correct-A, failed-A, and C sensor trajectories
- channel-level Random Forest feature-importance analysis
- detection of unusually large sample-to-sample jumps
- timing-robust 54-feature summaries using nine statistics across six channels
- PCA visualization of live A/B/C summary features
- leave-one-out Random Forest testing of the 54-feature representation
- feature-importance analysis for the timing-robust summaries

This notebook is part of the experimental history that motivated the later segmented 216-feature representation.
