#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time

class data_setup():
    def __init__(self):
        self.trainloader = []
        self.training_label = []

        self.trainloader_left = []
        self.training_label_left = []
        self.trainloader_right = []
        self.training_label_right = []

    def one_folder_img_trainloader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'

        print("Setting up data for neural network!")

        for img in os.listdir(DATADIR):
            try:
                action = self.parse(img)
                self.training_label.append(action)
                img_array = cv2.imread(os.path.join(DATADIR, img))  # <type 'numpy.ndarray'>
                img_resize = self.resize(img_array)  
                img_tensor = torch.from_numpy(img_resize)  # <class 'torch.Tensor')
                self.trainloader.append(img_tensor)
                break
            except Exception as e:
                pass

    def two_folder_img_trainloader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/'
        CATEGORIES = ['left', 'right']

        print("Setting up data for neural network!")

        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img)) 
                    img_resize = self.resize(img_array)
                    img_tensor = torch.from_numpy(img_resize)
                    if category == 'left':
                        action = self.parse(img)
                        self.training_label_left.append(action)
                        self.trainloader_left.append(img_tensor)
                    else:
                        action = self.parse(img)
                        self.training_label_right.append(action)
                        self.trainloader_right.append(img_tensor)
                    break
                except Exception as e:
                    pass

    def parse(self, string):
        ls = string.split('-')
        if len(ls) == 6:  # negative x and z coordinates
            x = '-' + ls[1]
            z = '-' + ls[3]
            return(x,z)
        elif len(ls) == 5:  # negative z coordinate
            x = ls[0]
            z = '-' + ls[2]
            return(x,z)
        else:
            x = ls[0]
            z = ls[1]
            return(x,z)

    def resize(self, img):
        scale_percent = 4 
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        #resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return(img)

if __name__ == "__main__":
    DATA = data_setup()

    start = time.perf_counter()
    DATA.one_folder_img_trainloader()
    print("One folder execution time: ", time.perf_counter() -start)
    print(DATA.training_label)
    
    start = time.perf_counter()
    DATA.two_folder_img_trainloader()
    print("Two folder execution time: ", time.perf_counter() - start)
    print(DATA.training_label_left)
    print(DATA.training_label_right)

    # count = 0
    # start = time.perf_counter()
    # for i in trainloader:
    #     count += 1
    # print(time.perf_counter() - start)
    # print(count)

    # VISUALIZING IMAGES
    # for data in trainloader:
    #     # print(data.size)
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break

    # for data in trainloader_left:
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break

    # for data in trainloader_right:
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break