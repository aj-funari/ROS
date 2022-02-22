#!/usr/bin/env python
import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class one_folder_setup():
    def __init__(self):
        self.trainloader = []
        self.training_label = []

    def folder_img_loader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'

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
        width = int(img.shape[1] / 32)  # x coordinate
        height = int(img.shape[0] / 24)
        # print("Resized image:", height, "x", width)
        dim = (width, height)
        # print(dim)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return(img)

    def num_batches(self, num_batch):
        batch_size = int(len(self.trainloader) / num_batch)
        epoch = []
        tmp = []
        rand_start = 0
        rand_end = batch_size
        check_rand_num = {}
 
        for i in range(num_batch):
            for j in range(batch_size):  
                while len(check_rand_num) < batch_size:
                    x = random.randrange(rand_start, rand_end)

                    if x not in check_rand_num.keys():  # if random number is not in dictorary, add random image
                        tmp.append(self.trainloader[x])  # append imgage to temporary list
                    
                    check_rand_num[x] = x  # add random number to dictionary 
            
            ### CHECK RANDOM NUMBERS
            # print("random numbers:", check_rand_num)
            # print("sorted number:", sorted(check_rand_num))
            
            rand_start += batch_size
            rand_end += batch_size

            epoch.append(tmp) # append list to create one epoch of data
            tmp.clear()  # clear temporary list for next batch
            check_rand_num.clear()  # clear dictionary for next batch

        ###  PRINT INFORMATION
        print("number of batches", num_batch)
        print("size of batch:", batch_size)
        print("length of epoch:", len(epoch))

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
    print("number of images: ", count)

    ### NUMBER OF LABELS
    count = 0
    for label in DATA.training_label:
        count += 1
    print("number of labels: ", count)

    # VISUALIZING IMAGES
    # for data in DATA.trainloader:
        # print(data.size)
        # plt.imshow(data, cmap='gray')
        # plt.show()
        # break

    ### VISUALIZING LABELS
    # print(DATA.training_label)

    ### BATCHES
    DATA.num_batches(10)
    DATA.num_batches(15)
    DATA.num_batches(20)
