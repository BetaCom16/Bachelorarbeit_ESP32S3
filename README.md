# Bachelorarbeit_ESP32S3-
Dieses Repository ist Bestandteil der Bachelorarbeit von Erik Pardey. Thema ist die automatisierte Bildklassifikation von Nutzpflanzen und Unkraut auf einem ressourcenbeschränkten eingebetteten System wie den ESP32-S3. Es beinhaltet den Programmcode für einen ESP32-S3 mit OV5640 Kameramodul und die dazugehörigen Jupyter Notebooks, um die Modelle für das maschinelle Lernen in Google Colab zu trainieren.

Stand 03.11.2026 wurde innerhalb des Projekts TensorFlow in der Version 2.20.0 und Keras in der Version 3.13.2 verwendet. Microsoft Visual Studio Code nutzte die Version 1.140, die Erweiterung ESP-IDF Version 4.1.0. Um Fehler zu vermeiden sollten beim Nachbau des Projekts die genannten Versionen der Komponenten berücksichtigt werden.

Die Jupyter-Notebooks können verwendet werden, um z.B. in Google Colab die Modelle MobileNetV1, MobileNetV2 und eine eigene Architektur zu trainieren. Dazu werden die Rohdaten vorbereitet und auf die Datensätze Training, Validierung und Test aufgeteilt. Nach dem Training lässt sich diese in das LiteRT-Format überführen, um auf dem ESP32-S3 ausführbar zu werden. Der anschließende Code beinhaltet das Auslesen von Metriken und der jeweiligen Confusion Matrix des FLOAT32- und INT8-Formats.

Innerhalb des Programmcodes für den ESP32-S3 wird zusammen mit dem seeed Erweiterungsboard ein Bild mittels der Kamera OV5640 aufgenommen, welches anschließend zugeschnitten und mittels Nearest-Neighbor-Interpolation in ein Bildformat skaliert wird, welches mit dem übereinstimmt, das auch für das Training verwendet wurde. Die anschließende Inferenz gibt eine Vorhersage über die am meisten erkannte Pflanzenart aus. Sobald der Programmcode auf den ESP32-S3 übertragen wurde, startet der Monitor und die Messungen können ausgeführt und ausgelesen werden.

Der verwendete Programmcode sollte für andere Architekturen als MobileNet und dem TinyCNN genutzt werden können, jedoch wurde dies nicht überprüft. 


# Jupyter Notebooks in Google Colab
## Datensatz importieren
Der jeweilige Datensatz wird als zip-Datei in Google Colab hochgeladen und extrahiert. Darin müssen sich die Bilder in den jeweiligen Ordnern befinden, die im weiteren Verlauf als Klassen identifiziert werden.
```
import zipfile

with zipfile.ZipFile("/content/dataset.zip", "r") as zip_ref:
  zip_ref.extractall("/content/")
```

## Datensatz vorbereiten
Der hochgeladene und extrahierte Datensatz muss nun für das Training vorbereitet werden. Der folgende Code definiert Eingangsgröße der Bilder, die Paketgröße und den Pfad zum Datensatz. Darauf folgend wird der Trainings- und Validierungsdatensatz definiert. Letzterer wird ein weiteres Mal für den Testdatensatz aufgeteilt. In den Codeblöcken wird definiert, dass die ursprünglichen Bilder zugeschnitten und mit der Nearest-Neighbor-Interpolation skaliert werden - so findet es auch auf dem ESP32-S3 statt.
```
import tensorflow as tf

bild_groesse = (96, 96)
batch_groesse = 32
datensatz_pfad = "/content/dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    datensatz_pfad,
    validation_split = 0.2,
    subset = "training",
    seed = 48,
    image_size = bild_groesse,
    batch_size = None,
    crop_to_aspect_ratio=True,
    interpolation="nearest"
)

rest_ds = tf.keras.utils.image_dataset_from_directory(
    datensatz_pfad,
    validation_split = 0.2,
    subset = "validation",
    seed = 48,
    image_size = bild_groesse,
    batch_size = None,
    crop_to_aspect_ratio=True,
    interpolation="nearest"
)

anzahl_rest = tf.data.experimental.cardinality(rest_ds).numpy()
haelfte = anzahl_rest // 2

val_ds = rest_ds.take(haelfte)
test_ds = rest_ds.skip(haelfte)

train_ds = train_ds.batch(batch_groesse)
val_ds = val_ds.batch(batch_groesse)
test_ds = test_ds.batch(batch_groesse)
```

## Training
Folgender Code definiert das Training für MobileNetV1. Angewendet wird hierbei Transfer Learning. Hierzu werden zu Beginn Parameter wie die Eingangsgröße und Größe des zu verwendenden Netzwerks definiert. Zusätzlich wird der ursprüngliche Klassifikationskopf des Netzwerks entfernt, um diesen für die eigene Klassifikation vorzubereiten. Zusätzlich wird die Technik der Data-Augmentation verwendet, ein Verfahren, um insbesondere bei kleineren Datensätzen mehr Variationen der begrenzten Daten zu ermöglichen. Das Training findet hierbei in 15 Epochen statt.
```
basis_modell_mobilenet_v1 = tf.keras.applications.MobileNet(
    input_shape=(96, 96, 3),
    alpha=0.25,
    include_top=False,
    weights="imagenet"
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

basis_modell_mobilenet_v1.trainable = False

modell_mobilenet_v1 = tf.keras.Sequential(
    [
    tf.keras.layers.Input(shape=(96, 96, 3)),
    data_augmentation,
    tf.keras.layers.Lambda(tf.keras.applications.mobilenet.preprocess_input),
    basis_modell_mobilenet_v1,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation="softmax")
    ], name="mobilenet_v1"
  )

modell_mobilenet_v1.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_mobilenet_v1 = modell_mobilenet_v1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)
```

## Quantisierung
Nach erfolgtem Training wird die Quantisierung durchgeführt. Dies wird empfohlen, um die begrenzten Ressourcen des ESP32-S3 zu berücksichtigen. Hierzu wurde ein repräsentativer Datensatz aus 100 Bildern definiert, um die darin befindlichen Float32-Parameter zu analysieren und optimal an das INT8-Format anzupassen.
```
def representative_dataset_mobilenet_v1():
    for bilder, _ in train_ds.unbatch().batch(1).take(100):
        yield [tf.cast(bilder, tf.float32)]

converter_mobilenet_v1 = tf.lite.TFLiteConverter.from_keras_model(modell_mobilenet_v1)
converter_mobilenet_v1.optimizations = [tf.lite.Optimize.DEFAULT]
converter_mobilenet_v1.representative_dataset = representative_dataset_mobilenet_v1
converter_mobilenet_v1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_mobilenet_v1.inference_input_type = tf.int8
converter_mobilenet_v1.inference_output_type = tf.int8

modell_mobilenet_v1_int8 = converter_mobilenet_v1.convert()
```

## Sichern und herunterladen
Mithilfe der beiden folgende Codeblöcke kann das quantisierte Modell in Colab gesichert und heruntergeladen werden. **Wichtig**: Der hochgeladene Datensatz sowie die gespeicherten Modelle sind nur temporär in Colab gesichert. Sobald die Verbindung zu Colab getrennt wurde, sind auch die darin befindlichen Daten nicht mehr vorhanden. Der Code bleibt jedoch erhalten.
```
with open("/content/pflanzenmodell_mobilenet1_int8.tflite", "wb") as datei:
    datei.write(modell_mobilenet_v1_int8)
```
```
from google.colab import files
files.download("/content/pflanzenmodell_mobilenet1_int8.tflite")
```

## Quantisierungsparameter auslesen
Mit dem folgenden Code können die Eigenschaften des Eingabetensors sowie dessen Quantisierungsparameter scale und zero_point ausgelesen werden. Diese Parameter sind für die korrekte Verarbeitung der Farbwerte auf dem ESP32-S3 essentiell. 
```
interpreter = tf.lite.Interpreter(model_content=modell_mobilenet_v1_int8)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]

print("Shape: ", input_details["shape"])
print("Datentyp:", input_details["dtype"])
print("Quantisierung:", input_details["quantization"])
```

## Metriken und Confusion Matrix des Float32-Modells auslesen
Bei Bedarf können die Metriken und die Confusion Matrix des jeweiligen Float32-Modells ausgelesen werden.
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

y_true = []
y_pred = []

for bilder, labels in test_ds:

    vorhersagen = modell_mobilenet_v1.predict(bilder, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(vorhersagen, axis=1))

print(classification_report(y_true, y_pred, target_names=["Boden", "Nutzpflanze", "Unkraut"]))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    display_labels=["Boden", "Nutzpflanze", "Unkraut"],
    cmap="Blues"
)

plt.title("MobileNetV1 Float32 Confusion Matrix")
plt.show()
```


## Metriken und Confusion Matrix des INT8-Modells auslesen
Bei Bedarf können auch die Metriken und die Confusion Matrix des quantisierten INT8-Modells ausgelesen werden. Dies ist nützlich, um analysieren zu können, ob und inwieweit sich die Metriken nach der Quantisierung verändert haben.
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

interpreter = tf.lite.Interpreter(model_content=modell_mobilenet_v1_int8)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

scale, zero_point = input_details["quantization"]

y_true = []
y_pred = []


for bilder, labels in test_ds:

    for bild, label in zip(bilder, labels):
        bild_int8 = np.round(bild.numpy() / scale + zero_point).astype(np.int8)
        bild_int8 = np.expand_dims(bild_int8,axis=0)
        interpreter.set_tensor(input_details["index"], bild_int8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        y_true.append(int(label.numpy()))
        y_pred.append(int(np.argmax(output)))

print(classification_report(y_true, y_pred,target_names=["Boden", "Nutzpflanze", "Unkraut"]))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    display_labels=["Boden", "Nutzpflanze", "Unkraut"],
    cmap="Blues"
)

plt.title("MobileNetV1 INT8 Confusion Matrix")
plt.show()
```

## Modelloperationen auslesen
Im Falle eines Architekturwechsels sollten die vom Modell benötigten Operatoren überprüft werden, um die begrenzten Ressourcen des ESP32-S3 zu berücksichtigen, was mit folgendem Code in Google Colab erreicht werden kann. Folgender Code liest beispielsweise die Operatoren für MobileNetV1 aus:

```
import tensorflow as tf

tf.lite.experimental.Analyzer.analyze(
    model_path="mobilenet_v1_int8.tflite"
)
```



# ESP32-S3 Programmcode
## Vorbereitung des Modells
Folgender Code wird benötigt, um das heruntergeladene Modell in ein C-Byte-Array zu überführen, damit der ESP32-S3 dieses ausführen kann. Dieser kann im Terminal eines Linux Betriebssystems ausgeführt werden. Die model_data.cc Datei wird im Verzeichnis "main" des Projekts abgelegt:
``` 
xxd -i MODELLNAME.tflite > model_data.cc
```

## Bibliotheken importieren
Folgende Bibliotheken müssen zu Beginn des Projekts importiert werden. Diese umfassen grundlegende C++- und spezifische ESP-Bibliotheken sowie den Inport von TensorFlow Lite Micro. Dies muss zuvor in das Projekt importiert werden:

```
idf.py add-dependency "espressif/esp-tflite-micro"
```

``` 
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
```

## GPIO's des ESP32-S3 deklarieren
Folgender Block definiert die standardmäßigen GPIO's des ESP32-S3:

```
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
```

## Definition der Tensor-Arena und Klassen
Dies definiert die Tensor-Arena und ein Array, welches die Klassen für die Vorhersagen beinhaltet.
```
constexpr size_t ARENA_SIZE = 2 * 1024 * 1024;

const char* KLASSEN[] = {
    "Boden",
    "Nutzpflanze",
    "Unkraut"
};
```

## Initialisierung der Kamera
Folgende Funktion konfiguriert die GPIO's des ESP32-S3 und legt Parameter für die Kamera wie z.B. Auflösung und RGB-Format fest. Zum Schluss wird true oder false zurückgegeben, je nachdem ob die Initialisierung der Kamera erfolgreich war oder nicht. Ohne eine erfolgreiche Initialisierung kann keine Bildaufnahme stattfinden.
```
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
```



## Vorverarbeitung des Bildes
Der folgende größere Codeblock ist für die korrekte Bildvorverarbeitung von großer Wichtigkeit. Ohne diesen werden die Bilder nicht so verarbeitet, wie sie auch beim Training verarbeitet wurden. Die Verarbeitung zwischen Training und Verarbeitung auf dem ESP32-S3 muss konsistent erfolgen. Der Block beginnt mit einem Zeiger auf die Struktur des Kamera-Framebuffers, welcher die Bildinformationen beinhaltet, und einen Zeiger auf den Eingabetensor, worin die vorbereiteten Bilddaten gespeichert werden. Mittels **crop_size** wird der niedrigste Wert der Bildhöhe und -breite ermittelt. Wird das aktuelle Projekt betrachtet, so ist von der 800x600 Pixel Auflösung der Wert 600 der niedrigste Wert, welcher in der Variable **crop_size** gespeichert wird.
```
void bild_vorbereiten(
    camera_fb_t* fb,
    TfLiteTensor* input
) {

    int crop_size = std::min(
        static_cast<int>(fb->width),
        static_cast<int>(fb->height)
    );
```

Die folgenden zwei Zeilen dienen dazu, um den Ausschnitt des Bildes zentriert festzuhalten. Das Bild ist 600 Pixel hoch, weshalb der Wert der Variable **y_offset** 0 beträgt. Die Variable **x_offset** definiert den horizontalen Versatz des Bildauschnitts.. Da diese bei der Aufnahme 800 Pixel beträgt, muss ein Ausschnitt definiert werden. Um diesen Ausschnitt zu zentrieren, müssen rechts und links davon jeweils 100 Pixel abgezogen werden, wodurch der 600 Pixel breite Ausschnitt zentriert platziert wird. 
```
    int x_offset = (static_cast<int>(fb->width) - crop_size) / 2;
    int y_offset = (static_cast<int>(fb->height) - crop_size) / 2;
```

Der folgende Block definiert zwei Schleifen, in denen die Nearest-Neighbor-Interpolation angewendet wird. Hierzu wird zu aller erst die Schleife für die Höhe definiert. Da das Eingangsbild so wie auch beim Training auf auf 96x96 skaliert werden soll, beginnt der Wert der **y-Variable** bei 0. Mittels der Variable **source_y** wird berechnet, welcher Pixel des ursprünglichen Kamerabildes für einen bestimmten Pixel des verkleinerten Bildes verwendet wird. Mit der in der x-Schleife befindlichen Variable **source_x** geschieht das gleiche, nur für die Breite und nicht für die Höhe. Dieser Vorgang wird als Nearest-Neighbor-Interpolation bezeichnet. 

**Beispiel**: Das Bild wird nach und nach zusammengesetzt und nun soll der Pixel bei y = 20 und x = 10 des 96x96 großen Bildes erstellt werden. Mithilfe der Formel von **source_y** und **source_x** ergibt sich folgendes: 
**source_y**: 0 + 20 * 600 / 96 = 125
**source_x**: 100 + 10 * 600 / 96 = 162

Das bedeutet, dass der Pixel mit den Koordinaten y = 20 und x = 10 die Farbwerte des Pixels aus dem Ursprungsbild aus der Zeile 125 und Position 162 erhält.

Direkt im Anschluss daran beginnt die Schleife für die Breite. Es lässt sich so vereinfacht vorstellen: Ein Pixel beginnt in der Zeile 0. Nun beginnt die x-Schleife damit, diese y-Zeile mit insgesamt 96 Pixeln zu füllen. Sobald dies erfolgte, springt die y-Schleife auf den Wert 1, woraufhin die x-Schleife auch diese Zeile vollständig mit Pixel füllt. Woher diese Pixel stammen, wird in kommenden Codeblöcken erläutert. Die Erklärung diente lediglich dem Verständnis, wie aus dem 600x600 Pixel Ausschnitt ein 96x96 Bild entsteht. Ohne den Ausschnitt von 600x600 würde das Bild verzerrt werden, wenn es von 800x600 auf 96x96 skaliert werden würde.

```

    for (int y=0; y < 96; y++) {
        int source_y =y_offset + y * crop_size / 96;
        for (int x=0; x < 96; x++) {

            int source_x = x_offset + x * crop_size / 96;
```

In diesem Projekt wurde RGB565 verwendet, um Speicherplatz einzusparen. Die Pixelwerte werden im Speicher des ESP32-S3 abgelegt. In der vorherigen Beispielrechnung konnte herausgefunden werden, dass der Pixel bei y = 20 und x = 10 Farbwerte aus dem Ursprungsbild aus Zeile 125 an Position 162 erhält. Diese Werte werden für das kommende Beispiel beibehalten. Im Speicher des Framebuffers beginnt der Pixel an der Position des Bytes 200324. Die Berechnung sieht dafür wie folgt aus: (125 * 800 + 162) * 2 = 200324. RGB565 setzt sich aus zwei Bytes zusammen, weshalb am Ende der Berechnung mit 2 multipliziert wird. Der Code dafür sieht wie folgt aus:
```
            size_t source_index = (static_cast<size_t>(source_y) * fb->width + source_x) * 2;
```

Folgende Codezeile ließt die zwei Bytes des jeweiligen Pixels aus. Hierfür wird **uint16_t** verwendet, ein Ganzzahltyp mit 16 Bit. Hierzu das erste Byte ausgelesenm welches mittels **<< 8** um 8 Bit nach links verschoben wird. Anschließend wird das zweite Byte ausgelesen, welches mit dem ersten Byte zu einem 16-Bit-Wert kombiniert wird.
```
            uint16_t pixel = (static_cast<uint16_t> (fb->buf[source_index]) << 8) | fb->buf[source_index + 1];
```

Im nächsten Schritt wird RGB565 in RGB888 umgewandelt, da das Modell RGB-Werte mit 8-Bit pro Kanal erwartet. Dazu wird der 16-Bit-Pixel in die drei Farbkanäle zerlegt. Bei RGB565 verwendet rot 5 Bit, grün 6 Bit und blau ebenfalls 5 Bit. Hierzu wurden drei Variablen für die drei Farbkanäle definiert. Rot wird um 11 Bitstellen nach rechts verschoben. Mittels der Bitmaske **0x1F**, werden nur die letzten 5 Bits behalten. Darauf folgt die Verschiebung von Grün um 5 Stellen nach rechts mit der Bitmaske 0x3F. Anschließend folgt Blau zusammen mit der Bitmaske 0x1F. Die Berechnung * 255 / 31 bzw. 63 skaliert den Bereich auf 0 - 255.
```
            int r =((pixel >> 11) & 0x1F) * 255 / 31;
            int g =((pixel >> 5) & 0x3F) * 255 / 63;
            int b = (pixel & 0x1F) * 255 / 31;
```

Folgende Zeile berechnet die Position im Eingabetensor. Dieser enthält 96 x 96 Pixel mit jeweils drei Farbwerten. **Beispiel**: Genommen werden wieder die Werte aus den vorherigen Beispielen, also y = 20 und x = 10. Daraus folgt die Berechnung: (20 * 96 + 10) * 3 = 5790. Die drei Farbkanäle liegen also an folgenden Indizes im Speicher: Index 5790 für Rot, 5791 für Grün und 5792 für Blau.
```
            size_t target_index = (y * 96 + x) * 3;
```

Im letzten Schritt müssen die Farbwerte in das INT8-Format umgewandelt werden, damit das Modell diese verarbeiten kann. Die Werte liegen bisher im Wertebereich zwischen 0 und 255 vor, was sich außerhalb des INT8-Bereichs befindet. Die verwendeten INT8-Modelle können nur Werte zwischen -128 und 127 verarbeiten. Nun kommen die zu Beginn erwähnten Quantisierungsparameter zum Einsatz. Bei der Quantisierung wurden der Skalierungsfaktor **scale = 1** und der Nullpunkt **zero_point = -128** verwendet. Durch die folgende Berechnung hat beispielsweise ein RGB-Wert von 128 den INT8-Wert 0 und 255 wären der INT8-Wert 127. Diese Werte werden in die Eingabetensor geschrieben. Im Anschluss daran endet die Vorverarbeitung für das jeweilige Bild.
```
            input->data.int8[target_index] = static_cast<int8_t>(r - 128);
            input->data.int8[target_index + 1] = static_cast<int8_t>(g - 128);
            input->data.int8[target_index + 2] = static_cast<int8_t>(b - 128);
        }
    }
}
```


## Einbinden des Modells
Die folgende Zeile bindet das Modell ein, indem mittels **GetModel** auf das Byte-Array des Modells in der Datei **model_data.cc** zugegriffen wird.
```
const tflite::Model* model = tflite::GetModel(pflanzenmodell_mobilenet1_int8_tflite);
```

## Definition der Tensor-Arena
Der folgende Block definiert die Tensor-Arena und legt den reservierten Speicher im PSRAM des ESP32-S3 ab.
```
    uint8_t* tensor_arena = static_cast<uint8_t*>(
            heap_caps_malloc(
                ARENA_SIZE,
                MALLOC_CAP_SPIRAM |
                MALLOC_CAP_8BIT
            )
        );
```
## Überprüfung der Tensor-Arena
Folgende Abfrage prüft, ob die Tensor-Arena angelegt werden konnte. Andernfalls wird **nullptr** zurückgegeben, was bedeutet, dass der Zeiger auf keinen gültigen Speicherbereich zeigt.
```
    if (tensor_arena == nullptr) {
        printf("Tensor-Arena Reservierung fehlgeschlagen\n");
        return;
    }
```


## Modelloperationen einbinden
Folgender Block definiert die Operationen, die das Modell verwendet.
```
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
```

## Definition des MicroInterpreters
Der MicroInterpreter führt die Inferenz aus. Hierzu definiert er das jeweilige Modell, die vorgegebenen Operationen und die Tensor-Arena mit dem zugewiesenen Speicherplatz. Dieser Block erstellt den Interpreter - die Inferenz wird hier noch nicht ausgeführt.
```
tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        ARENA_SIZE
    );
```


## Tensoren vorbereiten
Dieser Block bereitet die Tensoren vor, woraufhin auf den Eingabetensor in **input** zugegriffen werden kann.
```
if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors fehlgeschlagen!\n");
    return;
    }

TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);
```

## Ausgabe der genutzten Tensor-Arena
Folgende Zeile liest aus, wie viele Bytes innerhalb des reservierten Arenabereichs tatsächlich genutzt wurden.
```
printf("Tensor-Arena genutzt: %u Bytes\n", (unsigned) interpreter.arena_used_bytes());
```


## Kameraprüfung und -vorbereitung
Die erste Abfrage prüft, ob die Kamera korrekt initialisiert wurde. Andernfalls wird das Programm beendet. War die Initialisierung erfolgreich, werden zu Beginn des Programms einmalig 5 Bilder aufgenommen und verworfen. Dies wurde implementiert, um durch eine mögliche Startverzögerung der Kamera die Inferenz nicht zu verfälschen.
```
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
```



## Inferenz und Ausgabe
Hier startet eine Schleife und darin die Inferenz. Insgesamt werden 30 Messungen mit einer Pause von 5 Sekunden nach jeder Inferenz durchgeführt. Dieser Block holt das aufgenommene Bild aus dem Framebuffer und prüft, ob eine Bildaufnahme überhaupt stattfinden konnte. Anschließend wird das jeweilige Bild durch die weiter oben erklärte Funktion **bild_vorbereiten** verarbeitet.
```
for (int messung = 1; messung <= 30; messung++) {

        camera_fb_t* fb = esp_camera_fb_get();

        if (fb == nullptr) {
            printf("Bildaufnahme fehlgeschlagen\n");
            return;
        }

        bild_vorbereiten(fb, input);

        esp_camera_fb_return(fb);
```

Hier wird ein Zeitstempel gesetzt, die Inferenz ausgeführt und anschließend wird ein weiterer Zeitstempel gesetzt, der vom vorherigen abgezogen wird, wodurch die Inferenzlatenz gemessen werden konnte.
```

        int64_t start = esp_timer_get_time();

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Inferenz fehlgeschlagen\n");
            return;
        }

        int64_t dauer_ms = (esp_timer_get_time() - start) / 1000;
```

Hier wird überprüft, welche Vorhersage das Modell für die jeweilige Inferenz getroffen hat. Die Variable **beste_klasse** beginnt bei 0, was der Klasse Boden entspräche. Das Ergebnis der Inferenz beinhaltet für die drei Klassen Werte, die Vorhersagewerte. Diese werden den Indexen der Klasse zugewiesen. Index 0 = Boden, Index 1 = Nutzpflanze, Index 2 = Unkraut. Die Schleife prüft folgendes: Ist der ausgegebene Wert von Index 1, also von der Nutzpflanze, höher als vom Boden, so wird Index 1 als beste Klasse festgelegt. 

Anschließend wird noch weiter geprüft, ob der Wert aus der Inferenz für den Index 2, also für Unkraut, höher als für die bisher beste Klasse ist. Ist dies nicht der Fall, bleibt Nutzpflanze die beste Klasse, ansonsten wird erhält die Variable den Wert 2, da Unkraut als beste Klasse identifiziert wurde.
```

        int beste_klasse = 0;

        for (int i = 1; i < 3; i++) {
            if (output->data.int8[i] > output->data.int8[beste_klasse]) {
                beste_klasse = i;
            }
        }
```

Hier findet eine Dequantisierung statt. Die INT8-Ausgabewerte werden in Gleitkommazahlen umgewandelt und mit 100 multipliziert, um eine Prozentdarstellung zu erhalten.
```
        float v_boden =(output->data.int8[0] - output->params.zero_point) * output->params.scale * 100.0f;
        float v_nutzpflanze = (output->data.int8[1] - output->params.zero_point) * output->params.scale * 100.0f;
        float v_unkraut = (output->data.int8[2] - output->params.zero_point) * output->params.scale * 100.0f;
```

Der folgende Block gibt die Anzahl der Messungen, die beste Klasse, die Prozentzahlen der jeweiligen Klassen und die Inferenzlatenz aus. Die Abfrage am Ende prüft, ob alle 30 Messungen durchgeführt wurden. Wenn dies der Fall ist, wird das Programm beeendet, andernfalls beginnt der Vorgang mit der Bildverarbeitung und der Inferenz von vorne.
```
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
```

Hier wird der freie RAM- und PSRAM-Speicher sowie die reservierte und genutzte Tensor-Arena ausgegeben.
```
    size_t freier_ram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t freier_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    printf("\n--- RAM und PSRAM ---\n");
    printf("Freier RAM: %u Bytes\n", (unsigned)freier_ram);
    printf("Freier PSRAM: %u Bytes\n\n",(unsigned)freier_psram);

    printf("--- Tensor-Arena ---\n");
    printf("Tensor-Arena reserviert: %u Bytes\n",(unsigned)ARENA_SIZE);
    printf("Tensor-Arena genutzt: %u Bytes\n", (unsigned) interpreter.arena_used_bytes());
```



