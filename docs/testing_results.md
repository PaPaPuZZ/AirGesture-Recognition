# Testing Results

## Numbers 1–5

Fresh authoritative holdout:

```text
Overall: 21/25 = 84%
1: 4/5
2: 4/5
3: 5/5
4: 3/5
5: 5/5
```

## Initial 6–0 Model

```text
Overall: 13/25 = 52%
6: 5/5
7: 4/5
8: 3/5
9: 1/5
0: 0/5
```

Confusions included:
- 8 -> 7
- 9 -> 6 / 7
- 0 -> 6 / 7 / 9

## 105-Recording 6–0 Model

```text
Overall: 17/25 = 68%
6: 0/5
7: 4/5
8: 5/5
9: 5/5
0: 3/5
```

Digit 6 was misclassified as 9 three times and 0 twice.

## 125-Recording 6–0 Model

```text
Overall: 20/25 = 80%
6: 5/5
7: 3/5
8: 3/5
9: 5/5
0: 4/5
```

## Unified 0–9 Cross-Validation

Training rows: 215

Selected model: Logistic Regression

```text
OOF accuracy:           90.70%
OOF balanced accuracy:  90.91%
OOF macro F1:           91.28%
```

Per-class recall:
- 0: 83.3%
- 1: 90.0%
- 2: 100.0%
- 3: 95.0%
- 4: 95.0%
- 5: 80.0%
- 6: 91.4%
- 7: 93.3%
- 8: 85.0%
- 9: 96.0%

## Unified 0–9 Fresh Live Certification

Fresh holdout using the 215-recording unified Logistic Regression model:

```text
Overall: 15/30 = 50%

0: 0/3
1: 2/3
2: 2/3
3: 1/3
4: 0/3
5: 2/3
6: 0/3
7: 2/3
8: 3/3
9: 3/3
```

Mean winning confidence remained high for several incorrect classes, including:
- 0: 67.6%
- 3: 91.7%
- 4: 61.3%
- 6: 63.6%

This result shows a substantial gap between random/stratified cross-validation and fresh live generalization. The numeric pipeline therefore remains under active iteration.

## Next Iteration Strategy

The next phase will increase gesture diversity rather than only adding examples for one failed class at a time.

Planned approach:
- collect additional examples for every digit 0–9
- deliberately vary size, speed, orientation, start position, and natural drawing style
- keep each collection session separately identifiable
- preserve all fresh certification sets as holdout evidence rather than merging them into training
- retrain and recertify after each meaningful dataset expansion
- later add additional participants and evaluate participant-independent generalization

The long-term goal is a model that generalizes across both gesture variation and different users.
