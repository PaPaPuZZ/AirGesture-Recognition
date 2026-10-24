# MPU6050 Orientation and Axis Robustness Study

## Goal

Determine whether the current IMU recognition pipeline remains accurate when the pen is held or used in different orientations and writing planes, and determine which physical sensor-axis alignment is best for the final pen PCB.

These are two separate experiments and should not be mixed within one collection session.

## Experiment A — Usage Orientation Robustness

Keep the MPU6050 mounted in the same physical orientation relative to the pen. Change only how the pen is held or the plane in which the gesture is drawn.

Recommended initial conditions:

| Orientation ID | Description |
|---|---|
| `O0_NEUTRAL_FRONT` | Current neutral grip; writing in the normal front-facing plane |
| `O1_ROLL_CW_90` | Pen rolled approximately 90° clockwise about its long axis |
| `O2_ROLL_CCW_90` | Pen rolled approximately 90° counterclockwise |
| `O3_HORIZONTAL` | Gesture drawn approximately parallel to a tabletop |
| `O4_DIAGONAL_45` | Gesture drawn in an approximately 45° diagonal plane |

For the first pilot, collect 5 trials per digit for each of the first four conditions:

```text
4 orientations × 10 digits × 5 trials = 200 recordings
```

## Experiment B — Sensor Mount / Axis Alignment

Keep the writing plane and user grip as consistent as possible. Change only how the MPU6050 is physically aligned relative to the pen body.

Suggested mount configurations:

| Mount ID | Pen-long-axis alignment |
|---|---|
| `MOUNT_X_LONG` | MPU6050 X axis aligned with pen length |
| `MOUNT_Y_LONG` | MPU6050 Y axis aligned with pen length |
| `MOUNT_Z_LONG` | MPU6050 Z axis aligned with pen length |

This experiment helps decide how the sensor should be oriented on the final PCB.

## Metadata

Every recording should preserve:

- `participant_id`
- `session_id`
- `dominant_hand`
- `orientation_id`
- `writing_plane`
- `sensor_mount_id`
- `pen_axis_alignment`
- approximate `roll_deg`, `pitch_deg`, and `yaw_deg`
- `actual`
- `trial_number`

## Evaluation

Do not rely only on randomly shuffled cross-validation.

Use:

1. ordinary stratified CV as a reference;
2. GroupKFold by `orientation_id`;
3. leave-one-orientation-out evaluation;
4. later, GroupKFold by `participant_id` when other users contribute data.

The main deployment question is whether a model trained on several orientations can recognize a held-out orientation.

## Model Development Sequence

1. Evaluate the current 216-feature model unchanged.
2. Add multi-orientation training data.
3. If held-out-orientation performance remains poor, compare:
   - acceleration magnitude and gyro magnitude features;
   - start-orientation normalization;
   - gravity subtraction / orientation-normalized acceleration;
   - random 3D rotation augmentation;
   - a small 1D CNN or TCN on the raw 100 × 6 sequence.

## Collection Rule

Change only one experimental variable per session. For example, do not simultaneously rotate the sensor mount and change the writing plane, because the result would not reveal which change affected performance.
