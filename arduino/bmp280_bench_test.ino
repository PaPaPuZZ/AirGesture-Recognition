#include <Wire.h>
#include <Adafruit_BMP280.h>

Adafruit_BMP280 bmp;

// Arduino Mega I2C:
// SDA = pin 20
// SCL = pin 21
//
// Use the voltage required by your specific BMP280 breakout.
// Many breakout boards include a regulator/level shifting, but not all do.
// Do not assume 5 V unless the board explicitly supports it.

const unsigned long SAMPLE_INTERVAL_MS = 100;  // 10 Hz
unsigned long lastSample = 0;

float baselinePressure = NAN;
float baselineTemperature = NAN;

void setup() {
  Serial.begin(115200);

  while (!Serial) {
    delay(10);
  }

  Serial.println();
  Serial.println("BMP280_BENCH_TEST");
  Serial.println("Searching for BMP280...");

  bool found = bmp.begin(0x76);

  if (!found) {
    found = bmp.begin(0x77);
  }

  if (!found) {
    Serial.println("BMP280_NOT_FOUND");
    Serial.println("Check wiring, voltage, and I2C address.");
    while (1) {
      delay(100);
    }
  }

  // Good general-purpose settings for observing pressure changes.
  bmp.setSampling(
    Adafruit_BMP280::MODE_NORMAL,
    Adafruit_BMP280::SAMPLING_X2,   // temperature
    Adafruit_BMP280::SAMPLING_X16,  // pressure
    Adafruit_BMP280::FILTER_X16,
    Adafruit_BMP280::STANDBY_MS_63
  );

  Serial.println("BMP280_READY");
  Serial.println();
  Serial.println("Hold sensor/chamber untouched for ~5 seconds.");
  Serial.println("Then press/deform the chamber repeatedly.");
  Serial.println();
  Serial.println(
    "time_ms,pressure_pa,pressure_hpa,delta_pa,"
    "temperature_c,delta_temp_c"
  );

  // Collect a short baseline.
  const int baselineSamples = 50;
  double pressureSum = 0.0;
  double temperatureSum = 0.0;

  for (int i = 0; i < baselineSamples; i++) {
    pressureSum += bmp.readPressure();
    temperatureSum += bmp.readTemperature();
    delay(100);
  }

  baselinePressure =
    pressureSum / baselineSamples;

  baselineTemperature =
    temperatureSum / baselineSamples;
}

void loop() {
  unsigned long now = millis();

  if (now - lastSample < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSample = now;

  float pressurePa = bmp.readPressure();
  float pressureHpa = pressurePa / 100.0f;
  float temperatureC = bmp.readTemperature();

  float deltaPa =
    pressurePa - baselinePressure;

  float deltaTemp =
    temperatureC - baselineTemperature;

  Serial.print(now);
  Serial.print(",");

  Serial.print(pressurePa, 2);
  Serial.print(",");

  Serial.print(pressureHpa, 4);
  Serial.print(",");

  Serial.print(deltaPa, 2);
  Serial.print(",");

  Serial.print(temperatureC, 2);
  Serial.print(",");

  Serial.println(deltaTemp, 3);
}
