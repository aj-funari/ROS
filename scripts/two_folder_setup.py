#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time

class data_setup():
    def __init__(self):
        self.trainloader_left = []
        self.training_label_left = []
        self.trainloader_right = []
        self.training_label_right = []

    def folders_img_loader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/'
        CATEGORIES = ['left', 'right']
        # count_left = 0
        # count_right = 0

        print("Setting up data for neural network!")

        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img)) 
                    img_resize = self.resize(img_array)
                    img_tensor = torch.from_numpy(img_resize)
                    if category == 'left':
                        self.trainloader_left.append(img_tensor)
                        
                        # count_left += 1
                        # if count_left == 3:
                        #     break
                    else:
                        self.trainloader_right.append(img_tensor)

                        # count_right += 1
                        # if count_right == 3:
                        #     break
                        
                except Exception as e:
                    pass

    def parse_folders(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        # count = 0

        for label in os.listdir(DATADIR):
            # count += 1
            # if count == 10:
            #     break

            label = label.split('-') 
            if len(label) == 6:  # negative x and z coordinates
                x = '-' + label[1]
                z = '-' + label[3]
                action = [x,z]
                self.training_label.append(action)

            elif len(label) == 5:  # negative z coordinate
                x = label[0]
                z = '-' + label[2]
                action = [x,z]
                self.training_label.append(action)
            
            else:
                x = label[0]
                z = label[1]
                action = [x,z]
                self.training_label.append(action)

    def resize_img(self, img): 
        # Orignial image = 768x1024
        ### Rescale down to 32x32
        width = round(img.shape[1] / 32)
        height = round(img.shape[0] / 24)
        ### Rescale down to 28X28
        # width = int(round(img.shape[1] / 36.57142857))  # x coordinate
        # height = int(round(img.shape[0] / 27.42857143))  # y coordinate
        print("Resized image:", height, "x", width)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return(img)