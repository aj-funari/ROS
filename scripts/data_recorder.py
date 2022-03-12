#!/usr/bin/env python

import os
import cv2
import rospy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from CNN import ResNet
from CNN import block

img_channels = 3
num_classes = 10
net = ResNet(block, [3,4,6,3], img_channels, num_classes)

bridge = CvBridge()

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.left_image = None
        self.right_image = None
        self.node = rospy.init_node('listener', anonymous=True)
        self.cmd_vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
        self.img_left = rospy.Subscriber('/front/left/image_raw', Image, self.left_img_callback)
        self.img_right = rospy.Subscriber('/front/right/image_raw', Image, self.right_img_callback)
        self.count = 0
        self.training_data = []

    def format(self, string):
        msg = string.split()
        x = msg[2]
        z = msg[13]
        msg = x + '-' + z
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = msg + '-' + current_time + '-' + '.jpeg'
        return(msg)
    
    def cmd_callback(self, msg):
        self.count += 1
        print(self.count)
        filename = self.format(str(msg))
        self.data = filename
        # print(self.data)

        directory_left = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        os.chdir(directory_left)
        cv2.imwrite(self.data, self.left_image)
        # print("left image saved")

        directory_right = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/right'
        os.chdir(directory_right)
        cv2.imwrite(self.data, self.right_image)
        # print("right image saved")

    def left_img_callback(self, image):
        # print("I recieved left Image!")
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            self.left_image = cv2_img
            # Feed Image through Neural Network
            out = ResNet(cv2_img)
            self.training_data.append(out)

        except CvBridgeError as e:
            pass

    def right_img_callback(self, image):
        # print("I received right Image!")
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            self.right_image = cv2_img
            # print(self.right_image)
        except CvBridgeError as e:
            pass

if __name__ =='__main__':
    data_recorder()
    rospy.spin()

# subscribe to node, collect image, feed through neural network, send output to publisher