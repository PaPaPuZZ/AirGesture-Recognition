# Scripts

This directory separates the project workflow by purpose.

- `data_collection/`: live Arduino/MPU6050 recording scripts
- `training/`: model training and cross-validation
- `certification/`: fresh live holdout tests
- `diagnostics/`: debugging and failure analysis
- `utilities/`: dataset merge/validation helpers

## Current reference workflow

For numeric recognition, the current reference pipeline is:

1. `training/train_numbers_0_9_candidates.py`
2. `certification/certify_numbers_0_9_unified.py`

The older 1-5 and 6-0 scripts are retained as experiment history and to document how the final unified pipeline evolved.

## Repository paths

Reference scripts should resolve data, models, and results relative to the repository root rather than assuming every file sits in one local folder.
