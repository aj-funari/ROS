#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time

class one_folder_setup():
    def __init__(self):
        self.trainloader = []
        self.training_label = []

    def folder_img_loader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        count = 0

        print("Setting up data for neural network!")

        for img in os.listdir(DATADIR):
            try:
                img_array = cv2.imread(os.path.join(DATADIR, img))  # <type 'numpy.ndarray'>
                img_resize = self.resize(img_array)  
                img_tensor = torch.from_numpy(img_resize)  # <class 'torch.Tensor')
                self.trainloader.append(img_tensor)

            except Exception as e:
                pass 

    def parse_folder(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        # count = 0

        for label in os.listdir(DATADIR):
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

    def resize(self, img): 
        # Orignial image = 768x1024
        # Rescale down to 32x32
        width = img.shape[1] / 32  # x coordinate
        height = img.shape[0] / 24
        # print("Resized image:", height, "x", width)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return(img)

if __name__ == "__main__":
    DATA = one_folder_setup()

    ### CALL FUNCTIONS TO SETUP DATA
    DATA.folder_img_loader()
    DATA.parse_folder()

    ### EXECUTION TIME
    # start = time.perf_counter()
    # DATA.folder_img_loader()
    # print("One folder execution time: ", time.perf_counter() -start)

    ### NUMBER OF IMAGES
    count = 0
    for i in DATA.trainloader:
        count += 1
    print("Number of images: ", count)

    ### NUMBER OF LABELS
    count = 0
    for label in DATA.training_label:
        count += 1
    print("Number of labels: ", count)

    # VISUALIZING IMAGES
    # for data in DATA.trainloader:
        # print(data.size)
        # plt.imshow(data, cmap='gray')
        # plt.show()
        # break

    ### VISUALIZING LABELS
    # print(DATA.training_label)
