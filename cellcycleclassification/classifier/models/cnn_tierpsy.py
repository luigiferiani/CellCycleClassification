#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:45:02 2020

@author: lferiani

Store models

"""

import torch.nn as nn
import torch.nn.functional as F


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


class CNN_tierpsy_roi48_v2(nn.Module):
    """
    Same CNN we use for tierpsy worm/non worm, half the channels in each layer
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                # conv layer taking the output of the previous layer:
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))  # activation layer

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(256*3*3, 2))

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


class CNN_tierpsy_roi48_v3(nn.Module):
    """
    Same CNN we use for tierpsy worm/non worm,
    half the channels in each layer
    removed a conv/conv/maxpool
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(256*6*6, 2))

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


class CNN_tierpsy_roi48_v4(nn.Module):
    """
    Same as v3 but using one node only at end bc binary, needs bcewithlogits
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(256*6*6, 1))

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
        return x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)


class CNN_tierpsy_roi48_original(nn.Module):
    """
    Original Tierpsy network
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                # conv 0
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv 1
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv2
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # define fully connected layer:
        self.fc_layers_with_dropout = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            )

    def global_max_pooling2d(self, input_tensor):
        # this was missing from the pytorch adaptation of avelino's cnn
        # it's tensorflow's GlobalMaxPooling2D
        # makes the network robust against roi size change,
        # since it removes x,y dimensions
        out = F.max_pool2d(
            input_tensor, kernel_size=input_tensor.size()[-2:])
        return out

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.global_max_pooling2d(x)  # maximum of each feature map
        # flatten output for fully connected layer,
        # end up being batch-by-channels, removes two dimensions at the end
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers_with_dropout(x)  # fully connected layers
        x = x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)
        return x


class CNN_tierpsy_roi48_original_v2(nn.Module):
    """
    Original Tierpsy network,
    remove one conv-conv-maxpool layer,
    halve fc nodes
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                # conv 0
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv 1
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv2
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # define fully connected layer:
        self.fc_layers_with_dropout = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 32),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            )

    def global_max_pooling2d(self, input_tensor):
        # this was missing from the pytorch adaptation of avelino's cnn
        # it's tensorflow's GlobalMaxPooling2D
        # makes the network robust against roi size change,
        # since it removes x,y dimensions
        out = F.max_pool2d(
            input_tensor, kernel_size=input_tensor.size()[-2:])
        return out

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.global_max_pooling2d(x)  # maximum of each feature map
        # flatten output for fully connected layer,
        # end up being batch-by-channels, removes two dimensions at the end
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers_with_dropout(x)  # fully connected layers
        x = x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)
        return x


class CNN_tierpsy_roi48_original_v3(nn.Module):
    """
    Original Tierpsy network,
    remove one conv-conv-maxpool layer,
    halve channels in all conv layers
    fc nodes are 1/4 of original
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                # conv 0
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv 1
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv2
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # define fully connected layer:
        self.fc_layers_with_dropout = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 16),
            nn.Dropout(p=0.5),
            nn.Linear(16, 1),
            )

    def global_max_pooling2d(self, input_tensor):
        # this was missing from the pytorch adaptation of avelino's cnn
        # it's tensorflow's GlobalMaxPooling2D
        # makes the network robust against roi size change,
        # since it removes x,y dimensions
        out = F.max_pool2d(
            input_tensor, kernel_size=input_tensor.size()[-2:])
        return out

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.global_max_pooling2d(x)  # maximum of each feature map
        # flatten output for fully connected layer,
        # end up being batch-by-channels, removes two dimensions at the end
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers_with_dropout(x)  # fully connected layers
        x = x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)
        return x


class CNN_tierpsy_roi48_original_v4(nn.Module):
    """
    Original Tierpsy network,
    but change global max pooling with global avg pooling
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                # conv 0
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv 1
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv2
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # define fully connected layer:
        self.fc_layers_with_dropout = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            )

    def global_avg_pooling2d(self, input_tensor):
        # this was missing from the pytorch adaptation of avelino's cnn
        # it's tensorflow's GlobalMaxPooling2D
        # makes the network robust against roi size change,
        # since it removes x,y dimensions
        out = F.avg_pool2d(
            input_tensor, kernel_size=input_tensor.size()[-2:])
        return out

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.global_avg_pooling2d(x)  # maximum of each feature map
        # flatten output for fully connected layer,
        # end up being batch-by-channels, removes two dimensions at the end
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers_with_dropout(x)  # fully connected layers
        x = x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)
        return x


class CNN_tierpsy_roi48_original_v5(nn.Module):
    """
    From original Tierpsy network, add conv layers to conv2 and conv3 group
    """
    roi_size = 48

    # Class : 1: S phase, 0: non S phase
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                # conv 0
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv 1
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv2
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # conv3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # define fully connected layer:
        self.fc_layers_with_dropout = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            )

    def global_max_pooling2d(self, input_tensor):
        # this was missing from the pytorch adaptation of avelino's cnn
        # it's tensorflow's GlobalMaxPooling2D
        # makes the network robust against roi size change,
        # since it removes x,y dimensions
        out = F.max_pool2d(
            input_tensor, kernel_size=input_tensor.size()[-2:])
        return out

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.global_max_pooling2d(x)  # maximum of each feature map
        # flatten output for fully connected layer,
        # end up being batch-by-channels, removes two dimensions at the end
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers_with_dropout(x)  # fully connected layers
        x = x.squeeze()  # BCEwithlogitsloss wants same size as labels (1d)
        return x


class CNN_tierpsy_roi48_multiclass(nn.Module):
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
        self.fc_layers = nn.Sequential(nn.Linear(512*3*3, 5))

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