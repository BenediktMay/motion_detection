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

        self.bridge = CvBridge()

        # Topics can be changed in the .launch file
        self.camera_topic_src = 'image_in'
        self.camera_topic_dst = 'motion/image_raw'

        self.declare_parameter('x_threshold', 0)
        self.declare_parameter('y_threshold', 0)
        self.declare_parameter('binary_threshold', 100)
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

        # Apply binary thresholding to detect black objects on white background
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the closed binary image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find and mark the largest contour
        best_contour = None
        largest_area = 0
        best_center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            if area > largest_area:
                largest_area = area
                best_contour = contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    best_center = (cX, cY)
                else:
                    best_center = None

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

        if best_contour is not None:
            self.last_rectangle = best_contour
            self.last_rectangle_time = current_time
            self.prev_center = best_center
            object_detected = True
        elif self.last_rectangle is not None and (current_time - self.last_rectangle_time) < 0.25:
            best_contour = self.last_rectangle
            object_detected = True
        else:
            self.last_rectangle = None
            self.last_rectangle_time = 0

        # Draw filled contour and center point
        if object_detected and best_contour is not None:
            cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
            if best_center is not None:
                cv2.circle(frame, best_center, 5, (0, 0, 255), -1)
            if self.prev_center is not None and best_center is not None:
                cv2.line(frame, self.prev_center, best_center, (255, 0, 0), 2)
            self.get_logger().info(f"Black object detected, area: {largest_area}, center: {best_center}")
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
