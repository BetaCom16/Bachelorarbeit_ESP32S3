# Bachelorarbeit_ESP32S3-
Dieses Repository ist Bestandteil der Bachelorarbeit von Erik Pardey. Thema ist die automatisierte Bildklassifikation von Nutzfplanzen und Unkraut auf einem ressourcenbeschränkten eingebetteten System wie den ESP32-S3. Es beinhaltet den Programmcode für einen ESP32-S3 mit OV5640 Kameramodul und die dazugehörigen Jupyter Notebooks, um die Modelle für das maschinelle Lernen in Google Colab zu trainieren.

Verwendet wurde TensorFlow in der Version 2.20.0 und Keras in der Version 3.13.2 (Stand: 30.10.2026).

Die Juypter Notebooks können verwendet werden, um z.B. in Google Colab die Modelle MobileNetV1, MobileNetV2 und eine eigene Architektur zu trainieren. Dazu werden die Rohdaten vorbereitet und auf die Datensätze Training, Validierung und Test aufgeteilt. Nach dem Training lässt sich diese in das LiteRT-Format überführen, um auf dem ESP32-S3 ausführbar zu werden. Der anschließende Code beinhaltet das Auslesen von Metriken und der jeweiligen Confusion Matrix des FLOAT32- und INT8-Formats.

Innerhalb des Programmcodes für den ESP32-S3 wird ein Bild mittels Kamera aufgenommen, welches anschließend zugeschnitten und mittels Nearest-Neighbor-Interpolation in ein Bildformat skaliert wird, welches mit dem übereinstimmt, das auch für das Training verwendet wurde. Die anschließende Inferenz gibt eine Vorhersage über die am meisten erkannte Pflanzenart aus.

