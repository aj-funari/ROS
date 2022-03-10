#!/usr/bin/env python

import os
import cv2
import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt

class one_folder_setup():
    def __init__(self):
        self.trainloader = []
        self.training_label = []
        self.batch_epoch = []
        self.label_epoch = []

    def folder_img_loader(self):
        DATADIR = '/home/aj/catkin_ws/src/ros_gazebo/scripts/images/left'

        print("Setting up data for neural network!")

        for img in os.listdir(DATADIR):
            try:
                img_array = cv2.imread(os.path.join(DATADIR, img))  # <type 'numpy.ndarray'>
                img_resize = self.resize(img_array)
                img_tensor = torch.from_numpy(img_resize).float()  # <class 'torch.Tensor')
                tensor = img_tensor.reshape([1, 3, 224, 224])
                self.trainloader.append(tensor)
            
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

    def resize(self, img): # image input size = 768x1024
        dim = (224, 224) # rescale down to 224x224
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # print(img.shape)
        return(img)

    def rand_batches_labels (self, num_batch):
        batch_size = int(len(self.trainloader) / num_batch)
        batches_tmp = []
        labels_tmp = []
        rand_start = 0
        rand_end = batch_size
        check_rand_num = {}
 
        for i in range(num_batch):
            for j in range(batch_size):
                i = 1
                while len(check_rand_num) < batch_size:
                    x = random.randrange(rand_start, rand_end)
                    if x not in check_rand_num.keys():  # if random number is not in dictorary, add random image
                        check_rand_num[x] = x # add num to check rand num list 
                        batches_tmp.append(self.trainloader[x])  # append imgage to temporary list
                        labels_tmp.append(self.training_label[x]) # append label to temporary list
                        print("appended random image/label #", i)
                        print(self.training_label[x], "#", i)
                        i += 1
            
            ### CHECK RANDOM NUMBERS
            # print("random numbers:", check_rand_num)
            # print("sorted number:", sorted(check_rand_num))
            
            rand_start += batch_size
            rand_end += batch_size
            
            ### ADD BATCHES OF IMAGES TOGETHER
            self.batch_epoch += batches_tmp
            batches_tmp.clear()  # clear temporary list for next batch

            ### ADD BATCHES OF LABELS TOGETHER
            self.label_epoch += labels_tmp
            labels_tmp.clear()

            ### CLEAR CHECK RAND NUM LIST
            check_rand_num.clear()  # clear the dictionary for next batch


        ###  PRINT INFORMATION
        print("number of batches", num_batch)
        print("size of each batch in batch epoch:", batch_size)
        print("size of batch epoch:", len(self.batch_epoch))
        # print("size of each batch in label epoch:",len(self.label_epoch[0]) )  
        # print("size of label epoch:", len(self.label_epoch))  

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
    #     print(data.size)
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break

    ### VISUALIZING LABELS
    # print(DATA.training_label)

    ### BATCHES
    DATA.rand_batches_labels(10)
    # DATA.num_batches(15)
    # DATA.num_batches(20)
