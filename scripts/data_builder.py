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

    def one_folder_img_loader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'
        # count = 0

        print("Setting up data for neural network!")

        for img in os.listdir(DATADIR):
            try:
                img_array = cv2.imread(os.path.join(DATADIR, img))  # <type 'numpy.ndarray'>
                img_resize = self.resize(img_array)  
                img_tensor = torch.from_numpy(img_resize)  # <class 'torch.Tensor')
                self.trainloader.append(img_tensor)
                
                # count += 1
                # if count == 3:
                #     break
            except Exception as e:
                pass

    def two_folder_img_loader(self):
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

    def parse_one_folder(self):
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

    ### CALL FUNCTIONS TO SETUP DATA
    DATA.one_folder_img_loader()
    DATA.parse_one_folder()

    ### EXECUTION TIME
    # start = time.perf_counter()
    # DATA.one_folder_img_loader()
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
    for data in DATA.trainloader:
        # print(data.size)
        plt.imshow(data, cmap='gray')
        plt.show()
        break

    ### VISUALIZING LABELS
    print(DATA.training_label)
