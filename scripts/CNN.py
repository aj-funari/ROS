#!/usr/bin/env python

import re
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from one_folder_setup import one_folder_setup

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4  # the number of channels after the block is always 4 times what is was when it entered
        self.conv1 = nn.Conv2d(
            in_channels, 
            intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels, 
            intermediate_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels, 
            intermediate_channels * self.expansion,
            kernel_size=1, 
            stride=1, 
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample  # only for first block
                                                        # this will be a conv layer that we're going to do to the identity 
                                                        # mapping so that it's of the same shape later on in the layers
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):  # ResNet50 = [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers 
        self.layer1 = self.make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layer(
            block, layers[3], intermediate_channels=512, stride=2)  # 2048

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        # only case when we change the numbere of channels and stide is in the first block
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)  # changes the number of channels
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))  # in_channels = 256; out_challens = 64: 256 -> 64, 64*4 = 256 again 

        return nn.Sequential(*layers)  # * --> will unpack list so PyTorch knows each layers come after another

def ResNet50(img_channels=3, num_classes=1000):
    return(ResNet(block, [3,4,6,3], img_channels, num_classes))

def ResNet101(img_channels=3, num_classes=1000):
    return(ResNet(block, [3,4,23,3], img_channels, num_classes))

def ResNet152(img_channels=3, num_classes=1000):
    return(ResNet(block, [3,8,36,3], img_channels, num_classes))

def test():
    net = ResNet50(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 244, 244))
    print("ResNet50 shape:", y.size())
    print(("ResNet50 output:", y))

    net = ResNet101(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 224, 224))
    print("ResNet101 shape:", y.size())
    print(("ResNet101 output:", y))

    net = ResNet152(img_channels=3, num_classes=10)
    y = net(torch.randn(1, 3, 224, 224))
    print("ResNet152 shape:", y.size())
    print(("ResNet152 output:", y))

if __name__ == '__main__':
    ### TEST RESNET ARCHITECTURE
    # test()

    ### MOVE MODEL/COMPUTATIONS TO GPU
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')

    # with torch.no_grad():
    #     output = model(input_batch)

    ### CREATE VARIABLE FOR CLASS
    DATA = one_folder_setup()

    ### LOAD IMAGES
    DATA.folder_img_loader()

    ### NUMBER OF IMAGES IN TRAINING SET
    # print(len(DATA.trainloader))

    ### CREATE EPOCHS
    DATA.num_batches(10)
    # print(len(DATA.epoch))    # --> 10 
    # for batch in DATA.epoch:  
    #     print(len(batch))     # --> 0?

    ### FEED DATA THROUGH NEURAL NETWORK
    net = ResNet50(img_channels=3, num_classes=2)
    
    test_tensor = torch.randn(1, 3, 224, 224)  # color dimension, height, width
    # out = (net(test_tensor))
    # print(out, "\nTEST TENSOR THROUGH NEURAL NETWORK!")

    tensor = DATA.trainloader[0]  # [1, 3, 224 , 224]
    out = net(tensor)
    print(out, "\nROS IMAGE THROUGH NEURAL NETWORK!")

    count = 0 
    for batch in DATA.epoch:
        for image in batch:
            image = image.reshape(1, 3, 224, 224)
            print("neural net output:", net(image))
            count += 1
        print("batch size:", count)
