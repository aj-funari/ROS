#!/usr/bin/env python

import re
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from one_folder_setup import one_folder_setup

<<<<<<< HEAD
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
=======
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        ...
>>>>>>> e4c0ae91a8b40b21e74849a4f123261bf0fa3c4b


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

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
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
    test()

    ### MOVE MODEL/COMPUTATIONS TO GPU
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')

    # with torch.no_grad():
    #     output = model(input_batch)

    ### CREATE VARIABLE FOR CLASS
    DATA = one_folder_setup()

    ### VISUALIZING IMAGES
    DATA.folder_img_loader()
    # for data in DATA.trainloader:
    #     print(data.size)
    #     plt.imshow(data, cmap='gray')
    #     plt.show()
    #     break

    ### VISUALIZING LABELS
    # DATA.parse_folder()
    # for label in DATA.training_label:
        # print(label)
    
    ### NEURAL NETWORK   
<<<<<<< HEAD
    # model = Net(32*32*3, 2)
    # for tensor in DATA.trainloader:
    #     tensor = tensor.view(1, -1)
    #     # print("Flattened tensor:", tensor)
    #     tensor = tensor.to(torch.float32)
    #     out = model(tensor)
    #     print(out)

    ### FEED DATA THROUGH RESNET
    net = ResNet50(img_channels=3, num_classes=10)
    y = net(DATA.trainloader[0])
    print(y)
=======
    model = Net(32*32*3, 2)
    for tensor in DATA.trainloader:
        tensor = tensor.view(1, -1)
        # print("Flattened tensor:", tensor)
        tensor = tensor.to(torch.float32)
        out = model(tensor)
        print(out)
        
        
    ### Rewrite code below 
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)
>>>>>>> e4c0ae91a8b40b21e74849a4f123261bf0fa3c4b
