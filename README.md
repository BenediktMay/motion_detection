# Motion Detection ROS2 Node

Dieses Paket enthält einen ROS2-Node zur Erkennung und Verfolgung eines schwarzen Rechtecks in Kamerabildern. Die Erkennung erfolgt über einstellbares (statisches oder dynamisches) Thresholding und Konturanalyse mit OpenCV.

## Features
- Erkennung eines dunklen (schwarzen) Rechtecks im Kamerabild
- Markierung des Rechtecks im Farbbild und im Binärbild
- Anpassbare Threshold-Strategie (statisch oder dynamisch)
- Robuste Konturerkennung (4-6 Ecken, konvex, Flächenfilter)
- Nachverfolgung des Rechtecks mit "Cooldown"-Logik
- ROS2-Parameter für Schwellenwerte, Framerate, Rechteckgröße etc.

## Nutzung
1. **Abhängigkeiten installieren**
   - ROS2 Foxy/Humble o.ä.
   - Python-Pakete: `opencv-python`, `numpy`, `cv_bridge`, `rclpy`

2. **Node starten**
   ```bash
   ros2 run motion_detection motion_detection_node
   ```
   oder per Launchfile:
   ```bash
   ros2 launch motion_detection motion_detection_launch.py
   ```

3. **Parameter anpassen**
   - Threshold-Wert (`static_thresh`) direkt im Code
   - Rechteckgröße, Framerate etc. im Code oder via ROS2-Parameter

4. **Topics**
   - Farbbild mit Markierung: `/motion/image_raw`
   - Binärbild: `/motion/binary`

## Hinweise
- Für wechselnde Lichtverhältnisse kann das dynamische Thresholding (auskommentiert im Code) aktiviert werden.
- Die Rechteckerkennung ist auf dunkle, rechteckige Objekte optimiert.
- Das Paket ist als Beispiel/Template für einfache Bildverarbeitung mit ROS2 und OpenCV gedacht.

## Lizenz
MIT
