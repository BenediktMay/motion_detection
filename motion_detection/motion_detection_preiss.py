import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math

class MotionDetectionNode(Node):
    def __init__(self):
        super().__init__('motion_detection_node')

        # Bridge for OpenCV/ROS conversion
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('x_threshold', 10)
        self.declare_parameter('y_threshold', 10)
        self.declare_parameter('max_distance', 0.2)
        self.declare_parameter('binary_threshold', 64)
        self.declare_parameter('cooldown_frames', 10)
        self.declare_parameter('framerate', 10)
        self.declare_parameter('image_topic', '/camera/color/image_raw')

        # Load parameter values
        self.x_threshold = self.get_parameter('x_threshold').get_parameter_value().integer_value
        self.y_threshold = self.get_parameter('y_threshold').get_parameter_value().integer_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        self.binary_threshold = self.get_parameter('binary_threshold').get_parameter_value().integer_value
        self.cooldown_frames = self.get_parameter('cooldown_frames').get_parameter_value().integer_value
        self.framerate = self.get_parameter('framerate').get_parameter_value().integer_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # Subscribe and publish
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, 'motion/image_raw', 10)
        self.binary_pub = self.create_publisher(Image, 'motion/binary', 10)

        # Internal states
        self.prev_center = None
        self.cooldown_counter = 0
        self.bounding_box = None
        self.last_time = time.time()

        # Setup RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.depth_frame = None

        self.get_logger().info("Motion Detection Node initialized")

    def image_callback(self, msg):
        # Frame rate limiting
        now = time.time()
        if now - self.last_time < 1.0 / self.framerate:
            return
        self.last_time = now

        # Convert to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        radius = int(min(center) // 1)

        # Update RealSense depth frame
        frames = self.pipeline.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        if not self.depth_frame:
            self.get_logger().warn("No depth frame available")
            return

        dist = self.depth_frame.get_distance(center[0], center[1])
        output_frame = frame.copy()
        cv2.circle(output_frame, center, radius, (0, 0, 255), 2)

        if dist < self.max_distance:
            # Region of interest mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)

            if cv2.countNonZero(binary) > (binary.size / 2.5):
                binary = cv2.bitwise_not(binary)

            # Publish binary for debug
            binary_msg = self.bridge.cv2_to_imgmsg(binary, encoding='mono8')
            self.binary_pub.publish(binary_msg)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_center = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 800 or area > 3000:
                    continue

                inside = all(
                    np.hypot(x - center[0], y - center[1]) <= radius
                    for [[x, y]] in contour
                )
                if not inside:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_center = (cx, cy)

                circle_area = math.pi * radius ** 2
                x, y, w, h = cv2.boundingRect(contour)
                if (w * h) > (circle_area * 0.8):
                    continue

                if self.prev_center is not None:
                    if abs(cx - self.prev_center[0]) > self.x_threshold or abs(cy - self.prev_center[1]) > self.y_threshold:
                        self.cooldown_counter = self.cooldown_frames

                if self.cooldown_counter > 0:
                    self.bounding_box = (x, y, w, h)

                break  # process only the first valid contour

            if self.cooldown_counter > 0 and self.bounding_box is not None:
                x, y, w, h = self.bounding_box
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.cooldown_counter -= 1

            self.prev_center = current_center

        # Publish final image
        img_msg = self.bridge.cv2_to_imgmsg(output_frame, encoding='bgr8')
        self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
