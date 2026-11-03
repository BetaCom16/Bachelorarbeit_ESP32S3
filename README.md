# Bachelorarbeit_ESP32S3-
Dieses Repository ist Bestandteil der Bachelorarbeit von Erik Pardey. Thema ist die automatisierte Bildklassifikation von Nutzpflanzen und Unkraut auf einem ressourcenbeschränkten eingebetteten System wie den ESP32-S3. Es beinhaltet den Programmcode für einen ESP32-S3 mit OV5640 Kameramodul und die dazugehörigen Jupyter Notebooks, um die Modelle für das maschinelle Lernen in Google Colab zu trainieren.

Stand 03.11.2026 wurde innerhalb des Projekts TensorFlow in der Version 2.20.0 und Keras in der Version 3.13.2 verwendet. Microsoft Visual Studio Code nutzte die Version 1.140, die Erweiterung ESP-IDF Version 4.1.0. Um Fehler zu vermeiden sollten beim Nachbau des Projekts die genannten Versionen der Komponenten berücksichtigt werden.

Die Jupyter-Notebooks können verwendet werden, um z.B. in Google Colab die Modelle MobileNetV1, MobileNetV2 und eine eigene Architektur zu trainieren. Dazu werden die Rohdaten vorbereitet und auf die Datensätze Training, Validierung und Test aufgeteilt. Nach dem Training lässt sich diese in das LiteRT-Format überführen, um auf dem ESP32-S3 ausführbar zu werden. Der anschließende Code beinhaltet das Auslesen von Metriken und der jeweiligen Confusion Matrix des FLOAT32- und INT8-Formats.

Folgender Code wird benötigt, um das heruntergeladene Modell in ein C-Byte-Array zu überführen, damit der ESP32-S3 dieses ausführen kann. Dieser kann im Terminal eines Linux Betriebssystems ausgeführt werden. Diese model_data.cc Datei wird im Verzeichnis "main" des Projekts abgelegt:
``` xxd -i MODELLNAME.tflite > model_data.cc ```


Innerhalb des Programmcodes für den ESP32-S3 wird zusammen mit dem seeed Erweiterungsboard ein Bild mittels der Kamera OV5640 aufgenommen, welches anschließend zugeschnitten und mittels Nearest-Neighbor-Interpolation in ein Bildformat skaliert wird, welches mit dem übereinstimmt, das auch für das Training verwendet wurde. Die anschließende Inferenz gibt eine Vorhersage über die am meisten erkannte Pflanzenart aus. Sobald der Programmcode auf den ESP32-S3 übertragen wurde, startet der Monitor und die Messungen können ausgeführt und ausgelesen werden.

Der verwendete Programmcode sollte für andere Architekturen als MobileNet und dem TinyCNN genutzt werden können, jedoch wurde dies nicht überprüft. Im Falle eines Architekturwechsels sollten die vom Modell benötigten Operatoren überprüft werden, um die begrenzten Ressourcen des ESP32-S3 zu berücksichtigen, was mit folgendem Code in Google Colab erreicht werden kann. Folgender Code liest beispielsweise die Operatoren für MobileNetV1 aus:

```
import tensorflow as tf

tf.lite.experimental.Analyzer.analyze(
    model_path="mobilenet_v1_int8.tflite"
)
```


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
