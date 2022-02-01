#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_builder import data_setup
import time
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30*40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
    
    # How do we want data to pass through neural network
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # don't want to run rectified linear; 
        return(F.log_softmax(x, dim=1))

if __name__ == '__main__':
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
    # for data in DATA.trainloader:
        # print(data.size)
        # plt.imshow(data, cmap='gray')
        # plt.show()
        # break

    ### VISUALIZING LABELS
    # print(DATA.training_label)

    ### NEURAL NETWORK
    # net = Net()

    # X = torch.rand((28, 28))
    # X = X.view(-1, 28*28)
    # print(net(X))

    # img = training_data[0]
    # img = img.view(-1, 768*1024)
    # print(net(img))
