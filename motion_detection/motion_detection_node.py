import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image



class MotionDetectionNode(Node):
    def __init__(self):
        # Parameter für Rechteckgröße (händisch im Code änderbar)
        self.rect_width = 40   # Breite des grünen Rechtecks
        self.rect_height = 40  # Höhe des grünen Rechtecks
        self.rect_thickness = 4  # Dicke des grünen Rahmens

        # Offset für dynamischen Threshold (händisch anpassbar)
        self.dynamic_threshold_offset = 20
        # Mindestfläche (Pixelanzahl) für "dunkelste Region" (händisch anpassbar)
        self.min_dark_area = 300
        super().__init__('motiondetectionnode')

        self.bridge = CvBridge()

        # Topics can be changed in the .launch file
        self.camera_topic_src = 'image_in'
        self.camera_topic_dst = 'motion/image_raw'

        self.declare_parameter('x_threshold', 0)
        self.declare_parameter('y_threshold', 0)
        self.declare_parameter('binary_threshold', 100) #standard auf 100 gesetzt
        self.declare_parameter('framerate', 10)
        self.declare_parameter('cooldown_frames', 10)

        self.x_threshold = self.get_parameter('x_threshold').get_parameter_value().integer_value
        self.y_threshold = self.get_parameter('y_threshold').get_parameter_value().integer_value
        self.binary_threshold = self.get_parameter('binary_threshold').get_parameter_value().integer_value
        self.framerate = self.get_parameter('framerate').get_parameter_value().integer_value
        self.cooldown_frames = self.get_parameter('cooldown_frames').get_parameter_value().integer_value

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic_src,
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            self.camera_topic_dst,
            10
        )

        self.image_pub_binary = self.create_publisher(
            Image,
            'motion/binary',
            10
        )

        self.last_time = time.time()
        self.prev_center = None
        self.last_rectangle = None
        self.last_rectangle_time = 0

    def image_callback(self, msg):
        current_time = time.time()
        time_diff = current_time - self.last_time

        if time_diff < 1 / self.framerate:
            return

        self.last_time = current_time

        # Convert the incoming image to OpenCV format
        frame = cv2.resize(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), (853, 480))
        if frame is None:
            self.get_logger().error("Can't read frame")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # --- Dynamisches Thresholding auskommentiert ---
        # perc5_gray = np.percentile(gray, 5)
        # dynamic_thresh = perc5_gray + self.dynamic_threshold_offset
        # dynamic_thresh = np.clip(dynamic_thresh, 0, 150)
        # self.get_logger().info(f"Dynamic threshold used: {dynamic_thresh:.2f} (5%-percentile: {perc5_gray:.2f})")
        # _, binary_dyn = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
        # _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # white_dyn = np.count_nonzero(binary_dyn)
        # white_otsu = np.count_nonzero(binary_otsu)
        # max_white = int(0.6 * binary_dyn.size)
        # if 100 < white_dyn < max_white and white_dyn >= white_otsu:
        #     binary = binary_dyn
        # elif 100 < white_otsu < max_white:
        #     binary = binary_otsu
        # else:
        #     if white_dyn >= max_white:
        #         binary = binary_otsu
        #     else:
        #         binary = binary_dyn if white_dyn > white_otsu else binary_otsu

        # --- Statisches Thresholding ---
        static_thresh = 70  # <<--- HIER festen Wert anpassen
        self.get_logger().info(f"Static threshold used: {static_thresh}")
        _, binary = cv2.threshold(gray, static_thresh, 255, cv2.THRESH_BINARY_INV)

        # Morphologische Operationen
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Konturen finden
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0
        best_center = None
        best_bbox = None
        fallback_contour = None
        fallback_area = 0
        fallback_bbox = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80:
                continue
            approx = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            # Akzeptiere 4-6 Ecken, konvex, passendes Seitenverhältnis
            if (4 <= len(approx) <= 6) and cv2.isContourConvex(approx) and 0.3 < aspect_ratio < 3.0 and 80 < area < 40000:
                if area > best_area:
                    best_area = area
                    best_contour = approx
                    best_bbox = (x, y, w, h)
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        best_center = (cX, cY)
                    else:
                        best_center = None
            # Fallback: größte Fläche, falls kein Rechteck
            if area > fallback_area:
                fallback_area = area
                fallback_contour = contour
                fallback_bbox = (x, y, w, h)

        # Falls kein Rechteck gefunden, nimm größte Fläche
        if best_contour is None and fallback_contour is not None:
            best_contour = fallback_contour
            best_area = fallback_area
            best_bbox = fallback_bbox
            # Mittelpunkt der BoundingBox
            x, y, w, h = best_bbox
            best_center = (int(x + w/2), int(y + h/2))

        # Fill the largest contour in the binary image for publishing
        binary_for_publish = closed.copy()
        if best_contour is not None:
            cv2.drawContours(binary_for_publish, [best_contour], -1, 255, thickness=cv2.FILLED)

        # Publish the binary image
        binary_msg = self.bridge.cv2_to_imgmsg(binary_for_publish, encoding='mono8')
        self.image_pub_binary.publish(binary_msg)

        # Track and draw the object
        object_detected = False
        current_time = time.time()

        # Speichere das letzte Rechteck und dessen Mittelpunkt
        if best_bbox is not None:
            self.last_rectangle = best_bbox
            self.last_center = best_center
            self.last_rectangle_time = current_time
            object_detected = True
        # Wenn kein Rechteck erkannt, aber noch in Cooldown (0.5s), verwende das letzte
        elif self.last_rectangle is not None and (current_time - self.last_rectangle_time) < 1:
            best_bbox = self.last_rectangle
            best_center = self.last_center
            object_detected = True
        else:
            self.last_rectangle = None
            self.last_center = None
            self.last_rectangle_time = 0
            best_bbox = None
            best_center = None
            object_detected = False

        # Zeichne nur das einstellbare grüne Rechteck um den Mittelpunkt (ohne Tracking-Linie)
        if object_detected and best_bbox is not None:
            x, y, w, h = best_bbox
            x1 = int(best_center[0] - self.rect_width // 2)
            y1 = int(best_center[1] - self.rect_height // 2)
            x2 = int(best_center[0] + self.rect_width // 2)
            y2 = int(best_center[1] + self.rect_height // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=self.rect_thickness)
            self.prev_center = best_center
            self.get_logger().info(f"Black object detected, area: {best_area}, center: {best_center}")
        else:
            self.get_logger().debug("No black object detected")

        # Publish the processed frame with overlay
        processed_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(processed_msg)


def main():
    rclpy.init()
    node = MotionDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
