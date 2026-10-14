// TrackX AI - RX node (SKELETON, untested on hardware)
// Role: on trigger message, capture N samples and stream to the laptop over serial.
// NOTE: internal ESP32 ADC is too slow for MHz ultrasound; use an external ADC
//       or analog envelope detection in a real build. Protocol scaffold only.
#include <WiFi.h>
#include <esp_now.h>

#define ADC_PIN 34
#define N_SAMPLES 2000

volatile bool armed = false;
volatile uint32_t lastSeq = 0;
uint16_t buf[N_SAMPLES];

typedef struct { uint32_t seq; } TriggerMsg;

void onRecv(const uint8_t *mac, const uint8_t *data, int len) {
  if (len == sizeof(TriggerMsg)) {
    TriggerMsg m; memcpy(&m, data, sizeof(m));
    lastSeq = m.seq;
    armed = true;
  }
}

void setup() {
  Serial.begin(921600);
  analogReadResolution(12);
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init failed"); return; }
  esp_now_register_recv_cb(onRecv);
}

void loop() {
  if (!armed) return;
  armed = false;

  for (int i = 0; i < N_SAMPLES; i++) buf[i] = analogRead(ADC_PIN);  // TODO: external ADC

  // Simple text protocol: "WAVE,<seq>,<n>\n v0,v1,...\n"
  Serial.printf("WAVE,%u,%d\n", (unsigned)lastSeq, N_SAMPLES);
  for (int i = 0; i < N_SAMPLES; i++) {
    Serial.print(buf[i]);
    Serial.print(i < N_SAMPLES - 1 ? "," : "\n");
  }
}
