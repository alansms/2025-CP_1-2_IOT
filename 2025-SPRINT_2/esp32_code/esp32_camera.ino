#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Configurações de WiFi
const char* ssid = "SUA_REDE_WIFI";
const char* password = "SUA_SENHA_WIFI";

// Endereço do servidor para envio das imagens
const char* serverAddress = "http://192.168.0.100:8501/upload_image";

// Pinos da câmera para ESP32-CAM (AI-THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// LED Flash
#define FLASH_LED_PIN 4
bool flashState = false;

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Desabilita o detector de brown-out

  Serial.begin(115200);
  Serial.println("ESP32-CAM Cartridge Detection System");

  // Configuração do LED Flash
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Configurar e inicializar a câmera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Qualidade da imagem
  config.jpeg_quality = 10;  // 0-63 (menor é melhor)
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Iniciar com resolução maior para melhor detecção
  config.frame_size = FRAMESIZE_UXGA; // Resolução 1600x1200

  // Inicializar a câmera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Falha ao inicializar a câmera: 0x%x", err);
    return;
  }

  // Reduzir resolução para transmissão mais rápida
  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_VGA); // 640x480

  // Conectar ao WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi conectado");
  Serial.print("Endereço IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Verificar conexão WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Conexão WiFi perdida, reconectando...");
    WiFi.begin(ssid, password);
    delay(2000);
    return;
  }

  // Capturar imagem
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Falha ao capturar imagem");
    delay(1000);
    return;
  }

  // Ligar o flash para a captura
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(100);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Enviar imagem para o servidor
  sendImageToServer(fb->buf, fb->len);

  // Liberar o buffer da imagem
  esp_camera_fb_return(fb);

  // Aguardar antes da próxima captura
  delay(5000);
}

void sendImageToServer(const uint8_t * buf, size_t len) {
  HTTPClient http;
  Serial.println("Enviando imagem para o servidor...");

  http.begin(serverAddress);
  http.addHeader("Content-Type", "image/jpeg");

  int httpResponseCode = http.POST(buf, len);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("Resposta do servidor: " + response);
  } else {
    Serial.print("Erro no envio da imagem. Código de erro: ");
    Serial.println(httpResponseCode);
  }

  http.end();
}
