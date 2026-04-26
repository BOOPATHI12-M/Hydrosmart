#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>
#include <ctype.h>
#include <string.h>

// -------------------- USER CONFIG --------------------
const char* WIFI_SSID = "bm";
const char* WIFI_PASSWORD = "12345678";
const char* SERVER_BASE = "http://10.250.233.174:5000";

#define FLOW_SENSOR_PIN 4
#define BUTTON_PIN 12
#define DS18B20_PIN 5

// GSM wiring (ESP32 <-> GSM module):
// ESP32 TX (GSM_TX_PIN) -> GSM RX
// ESP32 RX (GSM_RX_PIN) -> GSM TX
// GND must be common between ESP32 and GSM module
#define GSM_RX_PIN 16
#define GSM_TX_PIN 17
const uint32_t GSM_BAUD = 9600;

// Relay channels: motor, normal stage, low-water alert, dry-run alert
#define RELAY_MOTOR_PIN 26
#define RELAY_STAGE1_PIN 25
#define RELAY_STAGE2_PIN 33
#define RELAY_STAGE3_PIN 32

#define BUZZER_PIN 27

// Most 4-channel relay boards are active LOW. Set true only if your board is active HIGH.
const bool RELAY_ACTIVE_HIGH = false;
const bool BUZZER_ACTIVE_HIGH = true;
const float PULSES_PER_LITER = 450.0f;
const unsigned long SEND_INTERVAL_MS = 2000;
const unsigned long DISPLAY_REFRESH_MS = 350;
const unsigned long BUTTON_DEBOUNCE_MS = 250;
const unsigned long MOTOR_START_GRACE_MS = 30000;
const char* DRY_STAGE_URL = "http://10.250.233.181:8081/bm";

const float LOW_FLOW_ALERT_MAX_LPM = 5.0f;
const float DRY_FLOW_THRESHOLD_LPM = 1.0f;

const uint8_t LCD_I2C_ADDR = 0x27;
const uint8_t LCD_COLS = 16;
const uint8_t LCD_ROWS = 4;
const uint32_t I2C_CLOCK_HZ = 50000;         // 50 kHz for better noise tolerance on long/noisy wiring
const uint32_t I2C_TIMEOUT_MS = 120;
const unsigned long LCD_RECOVER_INTERVAL_MS = 1000;
const uint16_t HTTP_CONNECT_TIMEOUT_MS = 900;
const uint16_t HTTP_RW_TIMEOUT_MS = 1100;
const uint8_t DS18B20_RESOLUTION_BITS = 11;
const unsigned long TEMP_SENSOR_RECOVER_INTERVAL_MS = 5000;
// -----------------------------------------------------

volatile uint32_t pulseCount = 0;
uint32_t lastPulseSnapshot = 0;

float totalLiters = 0.0f;
float flowRate = 0.0f;
float waterTemperatureC = 0.0f;
float motorCurrentA = 0.0f;
float motorVoltageV = 0.0f;

unsigned long lastSendMs = 0;
unsigned long lastDisplayMs = 0;
unsigned long lastButtonMs = 0;
unsigned long motorStartMs = 0;
unsigned long lastWiFiRetryMs = 0;

bool motorEnabled = false;
bool lastButtonReading = HIGH;
bool lastMotorRequested = false;
bool buttonMotorRequest = false;
bool serverMotorRequest = false;
bool dryRunTrip = false;
bool dryStageNotified = false;
int dryStageResponse = 0;
bool lcdOnline = false;
bool lastMotorRelayOn = false;
unsigned long lastLcdRecoverMs = 0;
unsigned long lastTempRecoverMs = 0;
uint8_t tempReadFailCount = 0;
bool ds18b20Detected = false;

char stageLabel[24] = "SYSTEM READY";

LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);
HardwareSerial GSMSerial(1);
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);
Adafruit_INA219 ina219;
bool ina219Ready = false;

char lastLcdRow0[17] = "";
char lastLcdRow1[17] = "";
char lastLcdRow2[17] = "";
char lastLcdRow3[17] = "";

void setStageLabel(const char* text) {
  strncpy(stageLabel, text, sizeof(stageLabel) - 1);
  stageLabel[sizeof(stageLabel) - 1] = '\0';
}

void invalidateLcdCache() {
  lastLcdRow0[0] = '\0';
  lastLcdRow1[0] = '\0';
  lastLcdRow2[0] = '\0';
  lastLcdRow3[0] = '\0';
}

bool isI2CDevicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission(true) == 0;
}

bool recoverLcd(bool showBootText = false) {
  lastLcdRecoverMs = millis();

  // Re-initialize I2C bus before LCD re-init in case motor noise disturbed the bus state.
  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Wire.setTimeOut(I2C_TIMEOUT_MS);

  if (!isI2CDevicePresent(LCD_I2C_ADDR)) {
    lcdOnline = false;
    return false;
  }

  lcd.init();
  lcd.backlight();
  lcd.clear();
  invalidateLcdCache();
  lcdOnline = true;

  if (showBootText) {
    lcd.setCursor(0, 0);
    lcd.print("AquaMind");
    lcd.setCursor(0, 1);
    lcd.print(" System Ready");
    delay(3000);
  }

  return true;
}

void serviceLcdHealth() {
  if (lcdOnline && isI2CDevicePresent(LCD_I2C_ADDR)) {
    return;
  }

  unsigned long now = millis();
  if (now - lastLcdRecoverMs >= LCD_RECOVER_INTERVAL_MS) {
    recoverLcd(false);
  }
}

void serviceDisplayTask();

bool checkGSMModule(uint32_t timeoutMs = 1500) {
  while (GSMSerial.available()) GSMSerial.read();
  GSMSerial.print("AT\r\n");

  char resp[64] = {0};
  size_t idx = 0;
  unsigned long start = millis();
  while (millis() - start < timeoutMs) {
    while (GSMSerial.available()) {
      char ch = (char)GSMSerial.read();
      if (idx < sizeof(resp) - 1) {
        resp[idx++] = ch;
        resp[idx] = '\0';
      }
      if (strstr(resp, "OK") != nullptr) {
        return true;
      }
    }
    delay(10);
  }
  return false;
}

bool getJsonStringField(const char* json, const char* key, char* out, size_t outSize) {
  if (!json || !key || !out || outSize == 0) return false;
  out[0] = '\0';

  char pattern[32];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);

  const char* keyPos = strstr(json, pattern);
  if (!keyPos) return false;

  const char* colon = strchr(keyPos + strlen(pattern), ':');
  if (!colon) return false;

  const char* p = colon + 1;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
    p++;
  }

  if (*p == '"') {
    p++;
    const char* endQuote = strchr(p, '"');
    if (!endQuote) return false;
    size_t len = (size_t)(endQuote - p);
    if (len >= outSize) len = outSize - 1;
    memcpy(out, p, len);
    out[len] = '\0';
    return true;
  }

  const char* end = p;
  while (*end && *end != ',' && *end != '}' && *end != '\r' && *end != '\n') {
    end++;
  }

  while (end > p && (*(end - 1) == ' ' || *(end - 1) == '\t')) {
    end--;
  }

  size_t len = (size_t)(end - p);
  if (len >= outSize) len = outSize - 1;
  memcpy(out, p, len);
  out[len] = '\0';
  return len > 0;
}

// -------------------- INTERRUPT (NOISE FILTERED) --------------------
void IRAM_ATTR onFlowPulse() {
  static uint32_t lastMicros = 0;
  uint32_t now = micros();

  if (now - lastMicros > 2000) {  // ignore noise
    pulseCount++;
    lastMicros = now;
  }
}

void setRelayPin(uint8_t pin, bool on) {
  if (RELAY_ACTIVE_HIGH)
    digitalWrite(pin, on ? HIGH : LOW);
  else
    digitalWrite(pin, on ? LOW : HIGH);
}

void setBuzzer(bool on) {
  if (BUZZER_ACTIVE_HIGH)
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
  else
    digitalWrite(BUZZER_PIN, on ? LOW : HIGH);
}

void printLcdRowIfChanged(uint8_t row, const char* text, char* cache) {
  if (!lcdOnline) {
    return;
  }

  if (strcmp(cache, text) == 0) {
    return;
  }

  lcd.setCursor(0, row);
  lcd.print(text);

  int len = strlen(text);
  for (int i = len; i < LCD_COLS; i++) {
    lcd.print(' ');
  }

  strncpy(cache, text, LCD_COLS);
  cache[LCD_COLS] = '\0';
}

bool isTemperatureValid(float tempC) {
  if (isnan(tempC)) {
    return false;
  }
  if (tempC == DEVICE_DISCONNECTED_C) {
    return false;
  }
  // 85.0C is a known DS18B20 power-up/default value before a proper conversion.
  if (fabsf(tempC - 85.0f) < 0.01f) {
    return false;
  }
  if (tempC < -55.0f || tempC > 125.0f) {
    return false;
  }
  return true;
}

void initTemperatureSensor(bool verboseLog = true) {
  ds18b20.begin();
  ds18b20Detected = (ds18b20.getDeviceCount() > 0);
  ds18b20.setWaitForConversion(true);
  ds18b20.setCheckForConversion(true);
  ds18b20.setResolution(DS18B20_RESOLUTION_BITS);

  if (!verboseLog) {
    return;
  }

  if (ds18b20Detected) {
    Serial.printf("DS18B20 detected (%u sensor)\n", ds18b20.getDeviceCount());
  } else {
    Serial.println("DS18B20 not detected");
  }
}

float readTemperatureC() {
  if (!ds18b20Detected) {
    return NAN;
  }

  float tempC = NAN;
  for (uint8_t attempt = 0; attempt < 2; attempt++) {
    ds18b20.requestTemperatures();
    tempC = ds18b20.getTempCByIndex(0);
    if (isTemperatureValid(tempC)) {
      return tempC;
    }
    delay(15);
  }

  return tempC;
}

float resolveTemperatureC() {
  float tempC = readTemperatureC();

  if (isTemperatureValid(tempC)) {
    tempReadFailCount = 0;
    return tempC;
  }

  if (tempReadFailCount < 255) {
    tempReadFailCount++;
  }

  if (tempReadFailCount == 1 || (tempReadFailCount % 5) == 0) {
    Serial.printf("Temperature read failed (%u), reporting N/A\n", tempReadFailCount);
  }

  unsigned long now = millis();
  if (now - lastTempRecoverMs >= TEMP_SENSOR_RECOVER_INTERVAL_MS) {
    lastTempRecoverMs = now;
    Serial.println("Reinitializing DS18B20 bus...");
    initTemperatureSensor(false);
  }

  return NAN;
}

float readCurrentA() {
  if (!ina219Ready) {
    return 0.0f;
  }

  float currentA = ina219.getCurrent_mA() / 1000.0f;
  if (fabsf(currentA) < 0.01f) {
    currentA = 0.0f;
  }
  return currentA;
}

float readVoltageV() {
  if (!ina219Ready) {
    return 0.0f;
  }

  float busV = ina219.getBusVoltage_V();
  float shuntV = ina219.getShuntVoltage_mV() / 1000.0f;
  float loadV = busV + shuntV;
  if (loadV < 0.0f) {
    loadV = 0.0f;
  }
  return loadV;
}

int notifyDryStage() {
  if (WiFi.status() != WL_CONNECTED) return 0;

  HTTPClient http;
  http.begin(DRY_STAGE_URL);
  http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
  http.setTimeout(HTTP_RW_TIMEOUT_MS);
  int code = http.GET();
  http.end();

  return code == HTTP_CODE_OK ? 1 : 0;
}

void applyStageControl() {
  bool motorRelayOn = false;
  bool stage1On = false;
  bool stage2On = false;
  bool stage3On = false;
  bool buzzerOn = false;
  motorEnabled = false;

  bool motorRequested = buttonMotorRequest || serverMotorRequest;
  unsigned long now = millis();

  if (motorRequested && !lastMotorRequested) {
    motorStartMs = now;
  }
  lastMotorRequested = motorRequested;

  if (!motorRequested) {
    setStageLabel("WAIT SIGNAL");
    dryRunTrip = false;
    dryStageNotified = false;
    dryStageResponse = 0;
  } else if ((now - motorStartMs) < MOTOR_START_GRACE_MS) {
    // Start motor first, then evaluate flow stages after grace time.
    setStageLabel("STARTING");
    motorEnabled = true;
    motorRelayOn = true;
    stage1On = true;
  } else if (dryRunTrip) {
    setStageLabel("DRY RUN ALERT");
    buzzerOn = true;
    stage3On = true;

    if (!dryStageNotified) {
      dryStageResponse = notifyDryStage();
      dryStageNotified = true;
      Serial.printf("Dry stage notify response: %d\n", dryStageResponse);
    }
  } else if (flowRate < DRY_FLOW_THRESHOLD_LPM) {
    // Dry motor protection: buzzer ON and motor relay auto OFF.
    setStageLabel("DRY RUN ALERT");
    buzzerOn = true;
    stage3On = true;
    dryRunTrip = true;
    buttonMotorRequest = false;
    serverMotorRequest = false;

    if (!dryStageNotified) {
      dryStageResponse = notifyDryStage();
      dryStageNotified = true;
      Serial.printf("Dry stage notify response: %d\n", dryStageResponse);
    }
  } else if (flowRate < LOW_FLOW_ALERT_MAX_LPM) {
    setStageLabel("LOW WATER ALERT");
    motorEnabled = true;
    motorRelayOn = true;
    stage2On = true;
    buzzerOn = true;
  } else {
    motorEnabled = true;
    motorRelayOn = true;
    dryStageNotified = false;
    dryStageResponse = 0;

    setStageLabel("NORMAL FLOW");
    stage1On = true;
  }

  setRelayPin(RELAY_MOTOR_PIN, motorRelayOn);
  setRelayPin(RELAY_STAGE1_PIN, stage1On);
  setRelayPin(RELAY_STAGE2_PIN, stage2On);
  setRelayPin(RELAY_STAGE3_PIN, stage3On);
  setBuzzer(buzzerOn);

  // Motor relay transitions are a high-noise moment; mark LCD as suspect and retry health soon.
  if (motorRelayOn != lastMotorRelayOn) {
    lastMotorRelayOn = motorRelayOn;
    lcdOnline = false;
    lastLcdRecoverMs = 0;
  }
}

void updateDisplay() {
  char line1[17];
  char line2[17];
  char line3[17];
  char line4[17];
  char tempText[8];
  char stageUpper[17];

  serviceLcdHealth();
  if (!lcdOnline) {
    return;
  }

  if (isnan(waterTemperatureC)) {
    strcpy(tempText, "--.-");
  } else {
    snprintf(tempText, sizeof(tempText), "%.1f", waterTemperatureC);
  }

  strncpy(stageUpper, stageLabel, sizeof(stageUpper) - 1);
  stageUpper[sizeof(stageUpper) - 1] = '\0';
  size_t stageLen = strlen(stageUpper);
  for (size_t i = 0; i < stageLen; i++) {
    stageUpper[i] = (char)toupper((unsigned char)stageUpper[i]);
  }

  snprintf(line1, sizeof(line1), "T:%5s M:%s", tempText, motorEnabled ? "ON" : "OFF");
  snprintf(line2, sizeof(line2), "FLOW:%5.1fL/M", flowRate);
  snprintf(line3, sizeof(line3), "I:%4.2fA V:%4.1f", motorCurrentA, motorVoltageV);
  snprintf(line4, sizeof(line4), "%-16.16s", stageUpper);

  // Avoid full-screen clear to reduce visible flicker/noise during motor operation.
  printLcdRowIfChanged(0, line1, lastLcdRow0);
  printLcdRowIfChanged(1, line2, lastLcdRow1);
  printLcdRowIfChanged(2, line3, lastLcdRow2);
  printLcdRowIfChanged(3, line4, lastLcdRow3);
}

void handleButton() {
  bool reading = digitalRead(BUTTON_PIN);
  if (reading == LOW && lastButtonReading == HIGH && (millis() - lastButtonMs) > BUTTON_DEBOUNCE_MS) {
    buttonMotorRequest = !buttonMotorRequest;
    // Give local button priority over any previous server manual command.
    serverMotorRequest = false;
    if (buttonMotorRequest) {
      dryRunTrip = false;
    }
    lastButtonMs = millis();
    Serial.printf("Button pressed -> ButtonRequest %s | ServerRequest %s\n",
                  buttonMotorRequest ? "ON" : "OFF",
                  serverMotorRequest ? "ON" : "OFF");
    // Apply motor/relay change immediately, no need to wait for next send cycle.
    applyStageControl();
  }
  lastButtonReading = reading;
}

bool parseRelayCommand(const char* json, bool& relayOn) {
  char relay[12];
  if (!getJsonStringField(json, "relay", relay, sizeof(relay))) {
    return false;
  }
  for (size_t i = 0; relay[i] != '\0'; i++) {
    relay[i] = (char)toupper((unsigned char)relay[i]);
  }

  if (strcmp(relay, "ON") == 0) {
    relayOn = true;
    return true;
  }
  if (strcmp(relay, "OFF") == 0) {
    relayOn = false;
    return true;
  }
  return false;
}

void serviceDisplayTask() {
  if (millis() - lastDisplayMs >= DISPLAY_REFRESH_MS) {
    lastDisplayMs = millis();
    updateDisplay();
  } else {
    serviceLcdHealth();
  }
}

// -------------------- WIFI --------------------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting WiFi");
  unsigned long t0 = millis();
  unsigned long lastDot = t0;

  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    serviceDisplayTask();
    if (millis() - lastDot >= 500) {
      Serial.print(".");
      lastDot = millis();
    }
    delay(10);
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi Failed!");
  }
}

// -------------------- SEND DATA --------------------
void sendData() {
  if (WiFi.status() != WL_CONNECTED) return;

  // Pulse calculation
  uint32_t snapshot;
  noInterrupts();
  snapshot = pulseCount;
  interrupts();

  uint32_t deltaPulses = snapshot - lastPulseSnapshot;
  lastPulseSnapshot = snapshot;

  float deltaLiters = deltaPulses / PULSES_PER_LITER;

  // Ignore noise
  if (deltaLiters < 0.001f) deltaLiters = 0.0f;

  totalLiters += deltaLiters;

  // Flow rate (L/min)
  flowRate = (deltaLiters * 60.0f) / (SEND_INTERVAL_MS / 1000.0f);

  // Sample additional analog sensors each telemetry cycle.
  waterTemperatureC = resolveTemperatureC();
  motorCurrentA = readCurrentA();
  motorVoltageV = readVoltageV();

  applyStageControl();

  if (isnan(waterTemperatureC)) {
    Serial.printf("Flow: %.2f L/min | Total: %.3f L | Temp: N/A | Current: %.2f A | Voltage: %.2f V | Stage: %s\n", flowRate, totalLiters, motorCurrentA, motorVoltageV, stageLabel);
  } else {
    Serial.printf("Flow: %.2f L/min | Total: %.3f L | Temp: %.2f C | Current: %.2f A | Voltage: %.2f V | Stage: %s\n", flowRate, totalLiters, waterTemperatureC, motorCurrentA, motorVoltageV, stageLabel);
  }

  // HTTP POST
  HTTPClient http;
  char url[96];
  snprintf(url, sizeof(url), "%s/update", SERVER_BASE);

  http.begin(url);
  http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
  http.setTimeout(HTTP_RW_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  char body[272];
  if (isnan(waterTemperatureC)) {
    snprintf(body, sizeof(body),
             "{\"liters\":%.3f,\"flow_rate\":%.3f,\"temperature\":null,\"current\":%.3f,\"voltage\":%.3f,\"stage\":\"%s\"}",
             totalLiters, flowRate, motorCurrentA, motorVoltageV, stageLabel);
  } else {
    snprintf(body, sizeof(body),
             "{\"liters\":%.3f,\"flow_rate\":%.3f,\"temperature\":%.2f,\"current\":%.3f,\"voltage\":%.3f,\"stage\":\"%s\"}",
             totalLiters, flowRate, waterTemperatureC, motorCurrentA, motorVoltageV, stageLabel);
  }

  serviceDisplayTask();
  int code = http.POST((uint8_t*)body, strlen(body));
  serviceDisplayTask();

  if (code > 0) {
    char resp[256] = {0};
    WiFiClient* stream = http.getStreamPtr();
    size_t offset = 0;
    unsigned long readStart = millis();

    while (millis() - readStart < HTTP_RW_TIMEOUT_MS && offset < sizeof(resp) - 1) {
      int avail = stream->available();
      if (avail > 0) {
        size_t toRead = (size_t)avail;
        size_t room = (sizeof(resp) - 1) - offset;
        if (toRead > room) toRead = room;
        size_t n = stream->readBytes(resp + offset, toRead);
        offset += n;
        readStart = millis();
      } else if (!stream->connected()) {
        break;
      }
      serviceDisplayTask();
      delay(1);
    }
    resp[offset] = '\0';

    char mode[16] = {0};
    getJsonStringField(resp, "mode", mode, sizeof(mode));
    for (size_t i = 0; mode[i] != '\0'; i++) {
      mode[i] = (char)toupper((unsigned char)mode[i]);
    }

    bool relayCmd;
    Serial.printf("Server: %s\n", resp[0] ? resp : "(empty)");
    Serial.printf("Backend mode: %s\n", mode[0] ? mode : "-");

    if (parseRelayCommand(resp, relayCmd)) {
      // Only allow server to START motor in explicit MANUAL mode.
      if (strcmp(mode, "MANUAL") == 0) {
         serverMotorRequest = relayCmd;
        if (serverMotorRequest) {
          dryRunTrip = false;
        }
        Serial.printf("Server MANUAL request -> %s\n", serverMotorRequest ? "ON" : "OFF");
        applyStageControl();
      } else {
        // In AUTO/unknown mode, never force ON from server response.
        if (!relayCmd) {
          serverMotorRequest = false;
          applyStageControl();
        }
        Serial.println("Server relay ignored (not MANUAL start)");
      }
    }

  } else {
    Serial.println("Server Error!");
  }

  http.end();
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);
  randomSeed((uint32_t)esp_random());

  GSMSerial.begin(GSM_BAUD, SERIAL_8N1, GSM_RX_PIN, GSM_TX_PIN);
  delay(300);
  if (checkGSMModule()) {
    Serial.println("OK)");
  } else {  
    Serial.println("GSM _OK)");
  }

  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Wire.setTimeOut(I2C_TIMEOUT_MS);

  initTemperatureSensor(true);
  ina219Ready = ina219.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Wire.setTimeOut(I2C_TIMEOUT_MS);
  if (!ina219Ready) {
    Serial.println("INA219 not found");
  } else {
    Serial.println("INA219 initialized");
  }

  pinMode(RELAY_MOTOR_PIN, OUTPUT);
  pinMode(RELAY_STAGE1_PIN, OUTPUT);
  pinMode(RELAY_STAGE2_PIN, OUTPUT);
  pinMode(RELAY_STAGE3_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  setRelayPin(RELAY_MOTOR_PIN, false);
  setRelayPin(RELAY_STAGE1_PIN, false);
  setRelayPin(RELAY_STAGE2_PIN, false);
  setRelayPin(RELAY_STAGE3_PIN, false);
  setBuzzer(false);

  if (!recoverLcd(true)) {
    Serial.println("LCD not detected at boot, auto-recovery enabled");
  }

  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), onFlowPulse, FALLING);

  connectWiFi();

  lastSendMs = millis();
}

// -------------------- LOOP --------------------
void loop() {
  handleButton();
  serviceDisplayTask();

  if (WiFi.status() != WL_CONNECTED) {
    // Keep trying WiFi without blocking LCD updates.
    if (millis() - lastWiFiRetryMs >= 5000) {
      lastWiFiRetryMs = millis();
      connectWiFi();
    }

    // Keep local state fresh even when offline.
    if (millis() - lastSendMs >= SEND_INTERVAL_MS) {
      lastSendMs = millis();
      waterTemperatureC = resolveTemperatureC();
      motorCurrentA = readCurrentA();
      motorVoltageV = readVoltageV();
      applyStageControl();
    }

    return;
  }

  if (millis() - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = millis();
    sendData();
  }
}
