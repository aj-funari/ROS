#!/usr/bin/env python

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_builder import data_setup
import time

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

    start = time.perf_counter()
    DATA.one_folder_img_trainloader()
    print("One folder execution time: ", time.perf_counter() -start)
    print(DATA.training_label)
    
    start = time.perf_counter()
    DATA.two_folder_img_trainloader()
    print("Two folder execution time: ", time.perf_counter() - start)
    print(DATA.training_label_left)
    print(DATA.training_label_right)

#net = Net()

#X = torch.rand((28, 28))
#X = X.view(-1, 28*28)
#print(net(X))

#img = training_data[0]
#img = img.view(-1, 768*1024)
#print(net(img))
