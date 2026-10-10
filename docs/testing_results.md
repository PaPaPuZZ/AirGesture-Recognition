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

Next: fresh unified 30-trial 0–9 certification.
