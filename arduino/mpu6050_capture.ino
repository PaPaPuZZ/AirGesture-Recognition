#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

const int NUM_SAMPLES = 100;
const unsigned long SAMPLE_INTERVAL_MS = 50;

void setup() {
  Serial.begin(115200);

  if (!mpu.begin()) {
    Serial.println("MPU6050_NOT_FOUND");
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
  Serial.println("MPU6050_READY");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'R') {
      recordGesture();
    }
  }
}

void recordGesture() {
  Serial.println("START");

  unsigned long recordingStart = millis();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    unsigned long targetTime =
      recordingStart +
      ((unsigned long)i * SAMPLE_INTERVAL_MS);

    while (millis() < targetTime) {
      // Wait until the next 50 ms sample point.
    }

    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t temp;

    mpu.getEvent(&accel, &gyro, &temp);

    // Adafruit library output:
    // acceleration: m/s^2
    // gyroscope: rad/s
    Serial.print(accel.acceleration.x, 6);
    Serial.print(",");
    Serial.print(accel.acceleration.y, 6);
    Serial.print(",");
    Serial.print(accel.acceleration.z, 6);
    Serial.print(",");
    Serial.print(gyro.gyro.x, 6);
    Serial.print(",");
    Serial.print(gyro.gyro.y, 6);
    Serial.print(",");
    Serial.println(gyro.gyro.z, 6);
  }

  unsigned long targetEnd =
    recordingStart +
    ((unsigned long)NUM_SAMPLES * SAMPLE_INTERVAL_MS);

  while (millis() < targetEnd) {
    // Keep the total gesture window at approximately 5 seconds.
  }

  Serial.println("END");
}
