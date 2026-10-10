# Hardware Setup and Debugging

## Arduino Mega + MPU6050

The primary prototype uses an Arduino Mega 2560 and MPU6050.

Important implementation details:
- I²C address observed: `0x68`
- Mega I²C pins: SDA = 20, SCL = 21
- Stable physical wiring and soldering improved reliability
- A pre-soldered sensor board was preferred after intermittent connections
- Mechanical mounting was explored using polycarbonate and standoffs

## Serial Protocol

```text
115200 baud
R -> START -> 100 x 6 sensor rows -> END
```

Useful macOS serial troubleshooting commands included:

```bash
lsof /dev/cu.usbmodem*
ls /dev/cu.usbmodem*
pkill -f python
```

## Units

Adafruit's MPU6050 library provides acceleration in m/s² and gyroscope values in rad/s. A previous live-pipeline error came from re-converting values that were already in SI units.
