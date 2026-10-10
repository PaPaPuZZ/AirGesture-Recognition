# Model Development History

## Early Anaconda / TinyML Work

Earlier work included:
- Kaggle air-writing dataset exploration
- Neuton TinyML experiments
- LSTM models in Anaconda/Jupyter
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
