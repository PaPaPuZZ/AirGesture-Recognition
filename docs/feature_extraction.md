# Temporal Feature Extraction

## Input

Each gesture is represented as a 100 × 6 matrix:

- Ax
- Ay
- Az
- Gx
- Gy
- Gz

## Segmentation

The 100 samples are divided into four consecutive 25-sample temporal segments.

## Features

For every channel in every segment:

1. mean
2. standard deviation
3. minimum
4. maximum
5. range / peak-to-peak
6. RMS
7. mean absolute first difference
8. maximum absolute first difference
9. sum of absolute first differences / total movement

Total:

```text
4 × 6 × 9 = 216 features
```

## Pipeline Consistency Lesson

Training and live prediction must use exactly the same feature definition and ordering. Earlier validation code produced the same dimensionality while using different statistics, which caused misleading live results.
