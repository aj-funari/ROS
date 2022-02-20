#!/usr/bin/env python

import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from one_folder_setup import one_folder_setup

class Net(nn.Module):
    def __init__(self, input_size, output_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')

    # with torch.no_grad():
    #     output = model(input_batch)

    ### Create variable for class
    DATA = one_folder_setup()

    ### VISUALIZING IMAGES
    DATA.folder_img_loader()
    # for data in DATA.trainloader:
    #     print(data.size)
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break

    ### VISUALIZING LABELS
    DATA.parse_folder()
    # for label in DATA.training_label:
        # print(label)
    
    ### NEURAL NETWORK   
    model = Net(32*32*3, 2)
    for tensor in DATA.trainloader:
        tensor = tensor.view(1, -1)
        # print("Flattened tensor:", tensor)
        tensor = tensor.to(torch.float32)
        out = model(tensor)
        print(out)
