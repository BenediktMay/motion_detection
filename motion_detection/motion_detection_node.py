import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image



class MotionDetectionNode(Node):
    def __init__(self):
        super().__init__('motiondetectionnode')
        # Parameter für Rechteckgröße (händisch im Code änderbar)
        self.rect_width = 40   # Breite des grünen Rechtecks
        self.rect_height = 40  # Höhe des grünen Rechtecks
        self.rect_thickness = 4  # Dicke des grünen Rahmens

        # Offset für dynamischen Threshold (jetzt als ROS-Parameter)
        self.declare_parameter('dynamic_threshold_offset', 50.0)
        self.dynamic_threshold_offset = self.get_parameter('dynamic_threshold_offset').get_parameter_value().double_value

        # Obergrenze für dynamischen Threshold (jetzt als ROS-Parameter)
        self.declare_parameter('dynamic_threshold_max', 120.0) #standardwert 120.0
        self.dynamic_threshold_max = self.get_parameter('dynamic_threshold_max').get_parameter_value().double_value

        # Statischer Threshold als ROS-Parameter
        self.declare_parameter('static_threshold', 68.0) # Standardwert 60.0
        self.static_threshold = self.get_parameter('static_threshold').get_parameter_value().double_value
        # Mindestfläche (Pixelanzahl) für "dunkelste Region" (händisch anpassbar)
        self.min_dark_area = 300

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
        self.prev_gray = None  # Für Bewegungserkennung

        self.image_pub_motion = self.create_publisher(
            Image,
            'motion/motion_mask',
            10
        )

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

        # Parameter zur Laufzeit auslesen (damit Änderungen sofort wirken)
        self.dynamic_threshold_offset = self.get_parameter('dynamic_threshold_offset').get_parameter_value().double_value
        self.dynamic_threshold_max = self.get_parameter('dynamic_threshold_max').get_parameter_value().double_value
        self.static_threshold = self.get_parameter('static_threshold').get_parameter_value().double_value

        # --- Bewegungserkennung (Frame-Differenz) ---
        motion_mask = None
        blue_bbox = None
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours_motion, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for cnt in contours_motion:
                area = cv2.contourArea(cnt)
                if area < 80:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if area > max_area:
                    max_area = area
                    blue_bbox = (x, y, w, h)
            # Zeichne blaue Bounding Box ins Farbbild (deaktiviert)
            # if blue_bbox is not None:
            #     bx, by, bw, bh = blue_bbox
            #     cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
        self.prev_gray = gray.copy()


        # --- Statisches Thresholding ---
        static_thresh = self.static_threshold
        self.get_logger().info(f"Static threshold used: {static_thresh}")
        _, binary = cv2.threshold(gray, static_thresh, 255, cv2.THRESH_BINARY_INV)

        # Morphologische Operationen
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Konturen finden und Rechteck suchen wie gehabt
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

        # --- Mask everything outside a center circle (erst jetzt) ---
        height, width = binary_for_publish.shape
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        mask = np.ones_like(binary_for_publish, dtype=np.uint8) * 255
        cv2.circle(mask, center, radius, 0, thickness=-1)
        binary_for_publish[mask == 255] = 255

        # Visualize: draw the circle on the binary image (optional)
        vis_binary = cv2.cvtColor(binary_for_publish, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis_binary, center, radius, (0, 0, 255), thickness=2)

        # Publish the binary image (masked and visualized)
        binary_msg = self.bridge.cv2_to_imgmsg(vis_binary, encoding='bgr8')
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
        elif self.last_rectangle is not None and (current_time - self.last_rectangle_time) < 2:
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
            # Prüfe, ob alle vier Ecken im Kreis liegen
            rect_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            def in_circle(pt):
                return (pt[0] - center[0])**2 + (pt[1] - center[1])**2 <= radius**2
            if all(in_circle(pt) for pt in rect_corners):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=self.rect_thickness)
            self.prev_center = best_center
            self.get_logger().info(f"Black object detected, area: {best_area}, center: {best_center}")
        else:
            self.get_logger().debug("No black object detected")

        # Draw the circle also on the output image
        cv2.circle(frame, center, radius, (0, 0, 255), thickness=2)

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
