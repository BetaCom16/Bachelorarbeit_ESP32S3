# Bachelorarbeit_ESP32S3-
Dieses Repository ist Bestandteil der Bachelorarbeit von Erik Pardey. Thema ist die automatisierte Bildklassifikation von Nutzfplanzen und Unkraut auf einem ressourcenbeschränkten eingebetteten System wie den ESP32-S3. Es beinhaltet den Programmcode für einen ESP32-S3 mit OV5640 Kameramodul und die dazugehörigen Jupyter Notebooks, um die Modelle für das maschinelle Lernen in Google Colab zu trainieren.

Stand 03.11.2026 wurde innerhalb des Projekts TensorFlow in der Version 2.20.0 und Keras in der Version 3.13.2 verwendet. Microsoft Visual Studio Code nutzte die Version 1.140, die Erweiterung ESP-IDF Version 4.1.0. Um Fehler zu vermeiden sollten beim Nachbau des Projekts die genannten Versionen der Komponenten berücksichtigt werden.

Die Juypter Notebooks können verwendet werden, um z.B. in Google Colab die Modelle MobileNetV1, MobileNetV2 und eine eigene Architektur zu trainieren. Dazu werden die Rohdaten vorbereitet und auf die Datensätze Training, Validierung und Test aufgeteilt. Nach dem Training lässt sich diese in das LiteRT-Format überführen, um auf dem ESP32-S3 ausführbar zu werden. Der anschließende Code beinhaltet das Auslesen von Metriken und der jeweiligen Confusion Matrix des FLOAT32- und INT8-Formats.

Folgender Code wird benötigt, um das heruntergeladene Modell in ein C-Byte-Array zu überführen, damit der ESP32-S3 dieses ausführen kann. Dieser kann im Terminal eines Linux Betriebssystems ausgeführt werden. Diese model_data.cc Datei wird im Verzeichnis "main" des Projekts abgelegt:
``` xxd -i MODELLNAME.tflite > model_data.cc ```


Innerhalb des Programmcodes für den ESP32-S3 wird zusammen mit dem seeed Erweiterungsboard ein Bild mittels der Kamera OV5640 aufgenommen, welches anschließend zugeschnitten und mittels Nearest-Neighbor-Interpolation in ein Bildformat skaliert wird, welches mit dem übereinstimmt, das auch für das Training verwendet wurde. Die anschließende Inferenz gibt eine Vorhersage über die am meisten erkannte Pflanzenart aus. Sobald der Programmcode auf den ESP32-S3 übertragen wurde, startet der Monitor und die Messungen können ausgeführt und ausgelesen werden.

