#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "DHT.h"
#include <Arduino.h>
#if defined(ESP32)
  #include <WiFi.h>
#elif defined(ESP8266)
  #include <ESP8266WiFi.h>
#endif
#include <Firebase_ESP_Client.h>

// OLED display configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 oled(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// DHT sensor configuration
#define DHTPIN 23
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Rain sensor configuration
#define POWER_PIN 19    // ESP32 GPIO19 that powers the rain sensor
#define AO_PIN 36       // ESP32 GPIO36 connected to AO pin of the rain sensor

// Buzzer configuration
#define BUZZER_PIN 5   // GPIO8 connected to the buzzer

// WiFi and Firebase configuration
#define WIFI_SSID "narzo 50 Pro 5G"
#define WIFI_PASSWORD "12345678"
#define API_KEY "{FIREBASE_API_KEY}"
#define DATABASE_URL "{DATABASE_URL}"

// Firebase objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
bool signupOK = false;

// Include Firebase helper libraries
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

// Variables for temperature, humidity, and rain values
String temperature;
String humidity;
String rainData;

void setup() {
  // Initialize Serial
  Serial.begin(115200);

  // Initialize DHT sensor
  dht.begin();

  // Set up rain sensor power pin
  pinMode(POWER_PIN, OUTPUT);
  analogSetAttenuation(ADC_11db);

  // Initialize Buzzer pin
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);  // Ensure the buzzer is off initially

  // Initialize OLED display
  if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    while (true);
  }
  delay(2000);          // wait for initializing
  oled.clearDisplay();  // clear display

  oled.setTextSize(2);       // text size
  oled.setTextColor(WHITE);  // text color

  // Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());
  Serial.println();

  // Firebase configuration
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback; // Monitor Firebase token generation

  // Sign up and initialize Firebase
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase signup successful");
    signupOK = true;
  } else {
    Serial.printf("Firebase signup error: %s\n", config.signer.signupError.message.c_str());
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  // Power on the rain sensor, wait, read the value, then power off
  digitalWrite(POWER_PIN, HIGH);        // Turn on rain sensor
  delay(10);                            // Wait for sensor to stabilize
  int rainValue = analogRead(AO_PIN);   // Read analog value from rain sensor
  digitalWrite(POWER_PIN, LOW);         // Turn off rain sensor
  Serial.print("Rain Sensor Value: ");
  Serial.println(rainValue);            // Print rain sensor value

  // Control buzzer based on rain sensor value
  if (rainValue < 4095) {
    digitalWrite(BUZZER_PIN, HIGH);  // Turn on buzzer
  } else {
    digitalWrite(BUZZER_PIN, LOW);   // Turn off buzzer
  }

  // Read DHT11 sensor data
  float humi = dht.readHumidity();
  float tempC = dht.readTemperature();

  // Check if data is valid
  if (isnan(humi) || isnan(tempC)) {
    temperature = "Failed";
    humidity = "Failed";
    rainData = "Failed";
  } else {
    temperature = String(tempC, 1) + "";
    humidity = String(humi, 1) + "";
    rainData = String(rainValue); // Display raw rain sensor value
  }

  // Update OLED display
  oledDisplayCenter(temperature, humidity, rainData);

  // Send data to Firebase if ready
  if (Firebase.ready() && signupOK) {
    if (Firebase.RTDB.setFloat(&fbdo, "/DHT_11/Humidity", humi)) {
      Serial.print("Humidity: ");
      Serial.println(humi);
    } else {
      Serial.print("Failed to send humidity data: ");
      Serial.println(fbdo.errorReason());
    }
    
    if (Firebase.RTDB.setFloat(&fbdo, "/DHT_11/Temperature", tempC)) {
      Serial.print("Temperature: ");
      Serial.println(tempC);
    } else {
      Serial.print("Failed to send temperature data: ");
      Serial.println(fbdo.errorReason());
    }
    
    if (Firebase.RTDB.setInt(&fbdo, "/Precipitation/Rain", rainValue)) {
      Serial.println("All data sent to Firebase");
    } else {
      Serial.print("Failed to send rain data: ");
      Serial.println(fbdo.errorReason());
    }
  }

  Serial.println("");
  delay(1000);  // Update every second
}

void oledDisplayCenter(String temperature, String humidity, String rainData) {
  oled.clearDisplay();  // clear display

  // Display temperature
  oled.setCursor(0, 0);
  oled.println("Temp:" + temperature);

  // Display humidity
  oled.setCursor(0, 20);
  oled.println("Humid:" + humidity);

  // Display rain data
  oled.setCursor(0, 40);
  oled.println("Rain:" + rainData);

  oled.display();
}