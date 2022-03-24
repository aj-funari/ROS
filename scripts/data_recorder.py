#!/usr/bin/env python

import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
import torch
import rospy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from one_folder_setup import one_folder_setup
from CNN_gpu import ResNet
from CNN_gpu import block

bridge = CvBridge()
setup = one_folder_setup()
move = Twist()

img_channels = 3
num_classes = 2
ResNet50 = ResNet(block, [3,4,6,3], img_channels, num_classes)

class data_recorder(object):

    def __init__(self):
        self.data = None
        self.left_image = None
        self.right_image = None
        self.tensor_x_z_actions = []
        self.count = 0
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('listener', anonymous=True)
        self.cmd_vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
        self.img_left = rospy.Subscriber('/front/left/image_raw', Image, self.left_img_callback)
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10) # definging the publisher by topic, message type
        self.rate = rospy.Rate(10)

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
        filename = self.format(str(msg))
        self.data = filename

        directory_left = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        os.chdir(directory_left)
        cv2.imwrite(self.data, self.left_image)
        
        self.count += 1
        print("Images saved:", self.count)  # print count of images saved while driving Jackal in Gazebo  

    def left_img_callback(self, image):
        # print("I recieved an image!")
        try:
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')  # returns array
            self.left_image = cv2_img

            # FEED IMAGES THROUGH NEURAL NETWORK
            # img_resize = setup.resize(cv2_img)
            # img_tensor = torch.from_numpy(img_resize)
            # img = img_tensor.reshape(1, 3, 224, 224)
            # # print(img.shape)
            # tensor_out = ResNet50(img)
            # self.tensor_x_z_actions.append(tensor_out)
            # print("Sent x-z actions to publisher!")

        except CvBridgeError as e:
            pass

    def publishMethod(self):    
        i = 0
        tmp = 1
        while not rospy.is_shutdown():
            # handle delay between subscriber and publisher
            if len(self.tensor_x_z_actions) == 0:
                # print("Successful pass!")
                pass
            else:
                if len(self.tensor_x_z_actions) >= tmp:  # publish actions only when action is sent from neural network output
                    print("x-z actions:", self.tensor_x_z_actions[i])
                    move.linear.x = self.tensor_x_z_actions[i][0][0]
                    move.linear.z = self.tensor_x_z_actions[i][0][1]
                    rospy.loginfo("Data is being sent") 
                    self.pub.publish(move)
                    self.rate.sleep()
                    i += 1
                    tmp += 1

if __name__ =='__main__':
    data_recorder()
    rospy.spin()
