# Author: Kyle Ma @ BCIL 
# Created: 05/16/2023
# Implementation of Automated Hematoma Segmentation

import torch
from torch import nn

# this model gives flexible leveling
class MV(nn.Module):
    def __init__(self, level):
        super(MV, self).__init__()

        # define how many levels we have
        self.level = level

        # first M1
        self.M1 = M1(1)

        # going down
        self.down_list = []

        for i in range(level):
            self.down_list.append(Down(32))

        # turn into pytorch layer list
        self.down_list = nn.ModuleList(self.down_list)

        # the bottom M2 since the last one only go right not down
        self.M2 = M2(32)
        
        # goes up the first up takes 64
        self.up_list = [Up(64)]

        # following ups takes 96
        for i in range(level - 1):
            self.up_list.append(Up(96))

        # turninto pytorch layer list
        self.up_list = nn.ModuleList(self.up_list)

        # M3 before output layer
        self.M3 = M3(96)


    def forward(self, input):
        # Named Accord to Architecture Graph in Paper
        # the _a are the inbetween variables

        # define level
        level = self.level

        # go through first M1
        down = self.M1(input)

        # keep the rights
        rights = []

        for i in range(level):
            
            # use downs to go deeper
            down, right = self.down_list[i](down)
            rights.append(right)

        # first Q to fed into the ups
        up = self.M2(down)

        # goes up
        for i in range(level):

            up = self.up_list[i](up)

            # after each step concate with the rights from M2
            up = torch.cat((up, rights[-(i+1)]), dim=1)

        # get the final output
        output = self.M3(up)

        return output



class Down(nn.Module):
    def __init__(self, input_channels):
        super(Down, self).__init__()

        self.M1 = M1(input_channels)
        self.M2 = M2(input_channels)
        self.Down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):

        down = self.M1(input)
        down = self.Down(down)
        right = self.M2(input)

        return down, right
    
class Up(nn.Module):
    def __init__(self, input_channels):
        super(Up, self).__init__()

        self.M1 = M1(input_channels)

    def forward(self, input):
        up = self.M1(input)
        up = nn.functional.interpolate(up, scale_factor=2, mode='bilinear')

        return up




# The M1 Moduel Two (3x3) Convolutional Layer with ReLU
class M1(nn.Module):
    def __init__(self, input_channels):
        super(M1, self).__init__()

        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, input):
        return self.DoubleConv(input)
    

# The M2 Module Three (3x3) Convolutional layers with ReLU + Dialation (1,2,4)
# Long skip layer concatenatingthe input of M2 with the output of M2
class M2(nn.Module):
    def __init__(self, input_channels):
        super(M2, self).__init__()

        self.TripleDia = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding="same", dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same", dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same", dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
            # TODO did they use batchnorm?
        )

    def forward(self, input):
        # TODO why Q2 and Q3 blocks are 32?
        return torch.cat((input, self.TripleDia(input)), dim=1)
    

# The M3 Module Two (3x3) Convo layers with ReLU and One (1x1) layer with Softmax
class M3(nn.Module):
    def __init__(self, input_channels):
        super(M3, self).__init__()

        self.TripleOut = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding="same"),

        )

    def forward(self, input):
        return self.TripleOut(input)
