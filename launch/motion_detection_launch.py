from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='motion_detection',
            executable='motion_detection_node',
            name='motion_detection_node',
            parameters=[
                {'x_threshold':0},
                {'y_threshold':0},
                {'max_dist':0.5},
                {'binary_threshold':100},
                {'framerate':10},
                {'cooldown_frames':10},
            ],
            remappings=[
                ('image_in','/camera/color/image_raw'), #/camera1/image_raw' for testing with usb camera
                ('depth_in', '/camera/depth/image_raw')
            ]
        )
        
    ])
