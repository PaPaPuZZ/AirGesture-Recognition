#include <Wire.h>
#include <Adafruit_BMP280.h>

Adafruit_BMP280 bmp;

const unsigned long SAMPLE_INTERVAL_MS = 100;  // 10 Hz
const int BASELINE_SAMPLES = 50;                // 5 seconds

unsigned long lastSample = 0;
float baselinePressurePa = 0.0f;

void calibrateBaseline() {
  double pressureSum = 0.0;

  // Keep the sensor/chamber untouched while this runs.
  for (int i = 0; i < BASELINE_SAMPLES; i++) {
    pressureSum += bmp.readPressure();
    delay(SAMPLE_INTERVAL_MS);
  }

  baselinePressurePa = pressureSum / BASELINE_SAMPLES;
}

void setup() {
  Serial.begin(115200);

  while (!Serial) {
    delay(10);
  }

  bool found = bmp.begin(0x76);

  if (!found) {
    found = bmp.begin(0x77);
  }

  if (!found) {
    // Keep this as a single numeric value so the Serial Plotter
    // is not confused by text output.
    while (1) {
      Serial.println(0.0);
      delay(500);
    }
  }

  bmp.setSampling(
    Adafruit_BMP280::MODE_NORMAL,
    Adafruit_BMP280::SAMPLING_X2,
    Adafruit_BMP280::SAMPLING_X16,
    Adafruit_BMP280::FILTER_X16,
    Adafruit_BMP280::STANDBY_MS_63
  );

  // Leave the sensor/chamber untouched for these first 5 seconds.
  calibrateBaseline();
}

void loop() {
  // Send b or B from the Serial Plotter/Monitor input box to re-zero.
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'b' || command == 'B') {
      calibrateBaseline();
    }
  }

  unsigned long now = millis();

  if (now - lastSample < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSample = now;

  float pressurePa = bmp.readPressure();
  float deltaPa = pressurePa - baselinePressurePa;

  // Output only the useful signal so the Serial Plotter auto-scales
  // around zero instead of around atmospheric pressure (~100,000 Pa).
  Serial.println(deltaPa, 2);
}
