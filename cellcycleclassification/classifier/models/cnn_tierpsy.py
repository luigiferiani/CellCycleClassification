#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:45:02 2020

@author: lferiani

Store models

"""

import torch.nn as nn


class CNN_tierpsy(nn.Module):
    """
    Same CNN we use for tierpsy worm/non worm
    """
    roi_size = 80

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                # conv layer taking the output of the previous layer:
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))  # activation layer

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(512*5*5, 2))

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.drop_out(x)
        # flatten output for fully connected layer, batchize,
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)  # pass  through fully connected layer
        # softmax activation function on outputs,
        # get probability distribution on output, all ouputs add to 1
        # x = F.softmax(x, dim=1)
        return x


class CNN_tierpsy_roi48(nn.Module):
    """
    Same CNN we use for tierpsy worm/non worm
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                # conv layer taking the output of the previous layer:
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))  # activation layer

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(512*3*3, 2))

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.drop_out(x)
        # flatten output for fully connected layer, batchize,
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)  # pass  through fully connected layer
        # softmax activation function on outputs,
        # get probability distribution on output, all ouputs add to 1
        # x = F.softmax(x, dim=1)
        return x
