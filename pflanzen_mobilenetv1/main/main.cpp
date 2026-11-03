#include <cstdio>
#include <cstdint>
#include <algorithm>
#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

constexpr size_t ARENA_SIZE = 2 * 1024 * 1024;

const char* KLASSEN[] = {
    "Boden",
    "Nutzpflanze",
    "Unkraut"
};

bool init_kamera() {
    camera_config_t config = {};
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_SVGA;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    return esp_camera_init(&config) == ESP_OK;
}


void bild_vorbereiten(
    camera_fb_t* fb,
    TfLiteTensor* input
) {

    int crop_size = std::min(
        static_cast<int>(fb->width),
        static_cast<int>(fb->height)
    );

    int x_offset = (static_cast<int>(fb->width) - crop_size) / 2;
    int y_offset = (static_cast<int>(fb->height) - crop_size) / 2;

    for (int y=0; y < 96; y++) {
        int source_y =y_offset + y * crop_size / 96;
        for (int x=0; x < 96; x++) {

            int source_x = x_offset + x * crop_size / 96;

            size_t source_index = (static_cast<size_t>(source_y) * fb->width + source_x) * 2;

            uint16_t pixel = (static_cast<uint16_t> (fb->buf[source_index]) << 8) | fb->buf[source_index + 1];

            int r =((pixel >> 11) & 0x1F) * 255 / 31;
            int g =((pixel >> 5) & 0x3F) * 255 / 63;
            int b = (pixel & 0x1F) * 255 / 31;

            size_t target_index = (y * 96 + x) * 3;

            input->data.int8[target_index] = static_cast<int8_t>(r - 128);
            input->data.int8[target_index + 1] = static_cast<int8_t>(g - 128);
            input->data.int8[target_index + 2] = static_cast<int8_t>(b - 128);
        }
    }
}

extern "C" void app_main() {

    printf("Starte Programm...\n");

    const tflite::Model* model = tflite::GetModel( pflanzenmodell_mobilenet1_int8_tflite);

    uint8_t* tensor_arena = static_cast<uint8_t*>(
            heap_caps_malloc(
                ARENA_SIZE,
                MALLOC_CAP_SPIRAM |
                MALLOC_CAP_8BIT
            )
        );

    if (tensor_arena == nullptr) {
        printf("Tensor-Arena Reservierung fehlgeschlagen\n");
        return;
    }

    tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        ARENA_SIZE
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors fehlgeschlagen!\n");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Tensor-Arena genutzt: %u Bytes\n", (unsigned) interpreter.arena_used_bytes());

    printf("Kamera starten\n");

    if (!init_kamera()) {
        printf("Kamera Fehler - Programm wird abgebrochen\n");
        return;
    }

    for (int i = 0; i < 5; i++) {
        camera_fb_t* fb = esp_camera_fb_get();

        if (fb != nullptr) {
            esp_camera_fb_return(fb);
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }

    for (int messung = 1; messung <= 30; messung++) {

        camera_fb_t* fb = esp_camera_fb_get();

        if (fb == nullptr) {
            printf("Bildaufnahme fehlgeschlagen\n");
            return;
        }

        bild_vorbereiten(fb, input);

        esp_camera_fb_return(fb);


        int64_t start = esp_timer_get_time();

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Inferenz fehlgeschlagen\n");
            return;
        }

        int64_t dauer_ms = (esp_timer_get_time() - start) / 1000;


        int beste_klasse = 0;

        for (int i = 1; i < 3; i++) {
            if (output->data.int8[i] > output->data.int8[beste_klasse]) {
                beste_klasse = i;
            }
        }

        float v_boden =(output->data.int8[0] - output->params.zero_point) * output->params.scale * 100.0f;
        float v_nutzpflanze = (output->data.int8[1] - output->params.zero_point) * output->params.scale * 100.0f;
        float v_unkraut = (output->data.int8[2] - output->params.zero_point) * output->params.scale * 100.0f;

        printf(
            "Messung %d/30 | "
            "Erkannt: %s | "
            "Boden: %.1f %% | "
            "Nutzpflanze: %.1f %% | "
            "Unkraut: %.1f %% | "
            "Inferenz: %lld ms\n",
            messung,
            KLASSEN[beste_klasse],
            v_boden,
            v_nutzpflanze,
            v_unkraut,
            dauer_ms
        );

        if (messung < 30) {
            vTaskDelay(pdMS_TO_TICKS(5000));
        }
    }


    size_t freier_ram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t freier_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    printf("\n--- RAM und PSRAM ---\n");
    printf("Freier RAM: %u Bytes\n", (unsigned)freier_ram);
    printf("Freier PSRAM: %u Bytes\n\n",(unsigned)freier_psram);

    printf("--- Tensor-Arena ---\n");
    printf("Tensor-Arena reserviert: %u Bytes\n",(unsigned)ARENA_SIZE);
    printf("Tensor-Arena genutzt: %u Bytes\n", (unsigned) interpreter.arena_used_bytes());
}