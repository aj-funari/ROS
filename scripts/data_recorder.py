#!/usr/bin/env python

import torch
import rospy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
import cv2 
import os
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from CNN import ResNet
from CNN import block
from one_folder_setup import one_folder_setup
import matplotlib.pyplot as plt

bridge = CvBridge()
setup = one_folder_setup()

img_channels = 3
num_classes = 2
ResNet50 = ResNet(block, [3,4,6,3], img_channels, num_classes)

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.left_image = None
        self.right_image = None
        self.node = rospy.init_node('listener', anonymous=True)
        self.cmd_vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
        self.img_left = rospy.Subscriber('/front/left/image_raw', Image, self.left_img_callback)
        self.count = 0
        self.tensor_x_z_actions = []

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
        # print(self.count)
        print(msg)
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
        # print("I recieved an image!")
        print("Sent x-z actions to publisher!")
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')  # returns array
            self.left_image = cv2_img

            # Feed image through neural network
            img_resize = setup.resize(cv2_img)
            img_tensor = torch.from_numpy(img_resize)
            img = img_tensor.reshape(1, 3, 224, 224)
            # print(img.shape)
            tensor_out = ResNet50(img)
            print("ResNet50 output:", tensor_out)
            self.tensor_x_z_actions.append(tensor_out)

        except CvBridgeError as e:
            pass

if __name__ =='__main__':
    data_recorder()
    rospy.spin()

    DATA = data_recorder()
    for i in range(len(DATA.training_data)):
        print(DATA.training_data[i])
