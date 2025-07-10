import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
import time


class MotionDetectionNode(Node):
    def __init__(self):
        super().__init__('motiondetectionnode')
        self.bridge = CvBridge()
        # Topics can be changed in the .launch file
        self.camera_topic_src = 'image_in'
        self.camera_topic_dst = 'motion/image_raw'

        self.declare_parameter('x_threshold',0)
        self.declare_parameter('y_threshold',0)
        self.declare_parameter('binary_threshold',100)
        self.declare_parameter('framerate',10)
        self.declare_parameter('cooldown_frames',10)

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
        self.prev_center = 0,0
        self.cooldown = 0

    def image_callback(self,msg):
        current_time = time.time()
        time_diff = current_time - self.last_time

        if time_diff < 1 / self.framerate:
            return
        self.last_time = current_time

        frame = cv2.resize(self.bridge.imgmsg_to_cv2(msg,'bgr8'),(853,480))
        if frame is None:
            self.get_logger().error(f"Can´t read frame")

        height,width = frame.shape[:2]
        center = (int(width/2),int(height/2))
        radius = int(min(center))

        output_frame = frame.copy()
        cv2.circle(output_frame,center,radius,(0,0,255),2)

        mask = np.zeros((height,width),dtype=np.uint8)
        cv2.circle(mask,center,radius,255,-1)
        masked_frame = cv2.bitwise_and(frame,frame,mask=mask)
        gray = cv2.cvtColor(masked_frame,cv2.COLOR_BGR2GRAY)
        _,binary = cv2.threshold(gray,self.binary_threshold,255,cv2.THRESH_BINARY)

        if cv2.countNonZero(binary) > (binary.size / 2.5):
            binary = cv2.bitwise_not(binary)

        self.image_pub_binary.publish(self.bridge.cv2_to_imgmsg(binary))                   
        contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_center = None
        bounding_box = None

        for contour in contours:
            area = cv2.contourArea(contour)
            # Check if Cotourarea is between something
            if area < 800 or area > 3200:
                continue

            inside = all(np.hypot(x - center[0], y - center[1]) <= radius for [[x,y]] in contour)
            if not inside:
                continue

            # Calc center of contour
            M = cv2.moments(contour)
            if M['m00']==0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_center = (cx,cy)

            circle_area = np.pi * (radius**2)
            x,y,w,h= cv2.boundingRect(contour)
            if (w*h) > (circle_area*0.8):
                continue
            moved = False
            if self.prev_center is not None:
                prev_x, prev_y = self.prev_center
                if abs(cx - prev_x) > self.x_threshold or abs(cy - prev_y) > self.y_threshold:
                    moved = True
            if moved:
                self.cooldown = self.cooldown_frames
            if self.cooldown > 0:
                bounding_box = (x,y,w,h)
            break

        if self.cooldown > 0 and bounding_box is not None:
            x,y,w,h = bounding_box
            cv2.rectangle(output_frame, (x,y),(x + w , y + h),(0,255,0),2)
            self.cooldown -= 1
        self.prev_center = current_center

        msg = self.bridge.cv2_to_imgmsg(output_frame,encoding='bgr8')
        self.image_pub.publish(msg)                   

        

def main():
    rclpy.init()
    node = MotionDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()

