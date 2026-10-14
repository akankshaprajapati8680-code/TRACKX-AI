// TrackX AI - TX node (SKELETON, untested on hardware)
// Role: fire the pulser circuit and notify the RX node over ESP-NOW.
#include <WiFi.h>
#include <esp_now.h>

#define PULSER_PIN 25          // placeholder: drives pulser circuit trigger
#define PULSE_INTERVAL_MS 500  // time between measurements

// TODO: replace with the RX node's MAC address
uint8_t rxAddress[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

typedef struct { uint32_t seq; } TriggerMsg;
TriggerMsg msg;
uint32_t seq = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PULSER_PIN, OUTPUT);
  digitalWrite(PULSER_PIN, LOW);

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init failed"); return; }

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, rxAddress, 6);
  peer.channel = 0;
  peer.encrypt = false;
  esp_now_add_peer(&peer);
}

void loop() {
  msg.seq = ++seq;
  esp_now_send(rxAddress, (uint8_t *)&msg, sizeof(msg));  // tell RX to arm capture
  delayMicroseconds(200);                                 // let RX arm

  digitalWrite(PULSER_PIN, HIGH);                          // short trigger to pulser
  delayMicroseconds(2);
  digitalWrite(PULSER_PIN, LOW);

  delay(PULSE_INTERVAL_MS);
}
