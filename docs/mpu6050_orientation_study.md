# MPU6050 Orientation and Board-Pose Robustness Study

## Goal

Determine whether the IMU recognition pipeline remains accurate when the MPU6050 is installed or used in different physical poses, and determine which orientations are realistic for the final pen design.

The first three collected datasets are **board-pose conditions**, not true X-, Y-, and Z-long-axis alignments.

## Collected Board-Pose Conditions

| Original label | Actual physical setup | Normalized orientation ID | Normalized mount ID |
|---|---|---|---|
| `AXIS_X_LONG` | MPU6050 board mounted vertically/on edge | `BOARD_FACE_VERTICAL` | `MOUNT_FACE_VERTICAL` |
| `AXIS_Y_LONG` | MPU6050 board horizontal and right-side up | `BOARD_FACE_UP` | `MOUNT_FACE_UP` |
| `AXIS_Z_LONG` | MPU6050 board horizontal and upside down | `BOARD_FACE_DOWN` | `MOUNT_FACE_DOWN` |

All three datasets remain useful. They test whether the same digit can be recognized when gravity and motion project differently onto the MPU6050 axes.

The raw source CSVs must remain unchanged. Use:

```bash
python scripts/utilities/normalize_number_orientation_metadata.py
```

to create corrected copies under:

```text
data/numbers/orientation_study/normalized/
```

The normalized copies preserve the original labels in `original_*` metadata columns.

## Important Interpretation Rule

The MPU6050 always records all six channels:

```text
Ax, Ay, Az, Gx, Gy, Gz
```

Rotating or flipping the board changes how gravity and physical pen motion project onto those channels. It does not cause the sensor to measure only one axis.

## True Axis-Alignment Study, If Needed Later

A true pen-axis study would require the physical MPU6050 axes to be aligned with the pen shaft:

| Mount ID | Required physical alignment |
|---|---|
| `MOUNT_X_LONG` | MPU6050 +X direction parallel to the pen shaft |
| `MOUNT_Y_LONG` | MPU6050 +Y direction parallel to the pen shaft |
| `MOUNT_Z_LONG` | MPU6050 +Z direction parallel to the pen shaft |

Because Z is perpendicular to the board face, a true `Z_LONG` configuration generally requires the board to be mounted on edge with its face perpendicular to the pen shaft.

Only collect these true axis-alignment conditions if they can be mounted securely and could realistically fit inside the final pen.

## Current Evaluation

Run:

```bash
python scripts/training/train_numbers_board_pose_robust.py
```

The training script compares:

1. `raw_axis_216`
   - the current 4-segment × 6-channel × 9-statistic representation;
2. `magnitude_72`
   - acceleration magnitude and gyroscope magnitude only;
3. `raw_plus_magnitude_288`
   - the original raw-axis features plus magnitude features.

For each feature set, the script evaluates:

- Logistic Regression;
- RBF SVM;
- Random Forest;
- ordinary stratified cross-validation;
- leave-one-board-pose-out validation.

The primary metric is leave-one-board-pose-out balanced accuracy, because the deployment question is whether the pen can recognize a gesture in a pose that was not used for training.

## Metadata

Every future orientation or mount recording should preserve:

- `participant_id`
- `session_id`
- `dominant_hand`
- `study_type`
- `orientation_id`
- `writing_plane`
- `sensor_mount_id`
- `pen_axis_alignment`
- approximate `roll_deg`, `pitch_deg`, and `yaw_deg` only when known
- `actual`
- `trial_number`

Do not invent roll, pitch, or yaw values when they were not measured.

## Collection Rule

Change only one experimental variable per session. Do not simultaneously change the sensor mount, writing plane, grip, and participant if the goal is to isolate orientation effects.

## Longer-Term Model Development

If board-pose generalization remains weak, compare:

- acceleration and gyroscope magnitude features;
- start-orientation normalization;
- gravity subtraction;
- rotation augmentation;
- a small 1D CNN or temporal convolutional network on the raw 100 × 6 sequence.
