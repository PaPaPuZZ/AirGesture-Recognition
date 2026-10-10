# Project Overview

## System Concept

The project is developing an IMU-based wearable gesture-recognition system for air-drawn letters, digits, and other hand motions, with a longer-term wearable glove implementation.

## Current Hardware

- Arduino Mega 2560
- MPU6050 IMU
- Planned second MPU6050 / second IMU
- BMP280 pressure sensor
- Planned microfluidic finger sensors
- Future custom PCB
- Wearable glove mechanical integration

## Current Data Acquisition Protocol

```text
Python sends: R
Arduino sends: START
Arduino sends: 100 rows of 6 values
Arduino sends: END
```

Each row contains `ax, ay, az, gx, gy, gz`.

Current units:
- acceleration: m/s²
- gyroscope: rad/s

Target timing:
- 100 samples
- 50 ms per sample
- approximately 5-second gesture window

## Current Technical Focus

Complete the unified numeric classifier, then return to M–Z before deeper multi-sensor and PCB integration.
