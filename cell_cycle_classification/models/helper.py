#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:40:22 2020

@author: lferiani
"""

import torch
import tables
import numpy as np
import pandas as pd

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CellsDataset(Dataset):

    LABELS_COLS = ['frame', 'x_center', 'y_center', 'label_is_S_phase']

    def __init__(self, hdf5_filename, which_set='train'):

        self.fname = hdf5_filename
        self.set_name = which_set + '_df'
        # get labels info
        ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        self.label_info = ann_df[self.LABELS_COLS]
        self.roi_size = 80  # size we want to train on
        with tables.File(self.fname, 'r') as fid:
            self.frame_height = fid.get_node('/full_data').shape[1]
            self.frame_width = fid.get_node('/full_data').shape[2]

        # any transform?
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            ])

    def __len__(self):
        return len(self.label_info)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # I could just use index because img_row_id is the same as the index of
        # label_info, but just in case we ever want to shuffle...
        label_info = self.label_info.iloc[index]
        # read images from disk
        roi_data = self._get_roi(label_info)

        # shift_and_normalize wants a float, and pytorch a single, use single
        img = roi_data.astype(np.float32)
        # normalize
        img = self.img_normalise(img)
        # fix dimensions for pytorch
        img = img[None, :, :]
        # make a tensor
        img = torch.from_numpy(img)
        # apply transformation
        img = self.transform(img)

        # read labels too
        labels = label_info['label_is_S_phase']
        labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
        labels = torch.from_numpy(labels)

        return img, labels

    # internal function to get a ROI's pixel data
    def _get_roi(self, info):

        hrs = self.roi_size // 2
        frame_number = info['frame']
        xc = int(info['x_center'])
        yc = int(info['y_center'])
        min_row = yc - hrs
        max_row = yc + hrs
        min_col = xc - hrs
        max_col = xc + hrs

        # roi is too small to be out of bounds at both ends
        pad_top = 0
        pad_bottom = 0
        if min_row < 0:
            pad_top = -min_row
            min_row = 0
        elif max_row > self.frame_height:  # end is excluded anyway
            pad_bottom = max_row - self.frame_height
            max_row = self.frame_height

        # roi is too small to be out of bounds at both ends
        pad_left = 0
        pad_right = 0
        if min_col < 0:
            pad_left = -min_col
            min_col = 0
        elif max_col > self.frame_width:  # end is excluded anyway
            pad_right = max_col - self.frame_width
            max_col = self.frame_width

        # get roi data
        with tables.File(self.fname, 'r') as fid:
            roi_data = fid.get_node(
                '/full_data')[
                    frame_number,
                    min_row:max_row,
                    min_col:max_col].copy()

        # pad if necessary
        roi_data = np.pad(
            roi_data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant')

        return roi_data

    def img_normalise(self, img):
        img -= img.mean()
        img /= img.std()
        return img


class ConvNet(nn.Module):
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
        x = nn.functional.softmax(x, dim=1)
        return x



# %%

if __name__ == "__main__":

    from pathlib import Path

    # where are things?
    work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    dataset_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'

    # parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 64

    # create datasets
    train_data = CellsDataset(dataset_fname, which_set='train')
    val_data = CellsDataset(dataset_fname, which_set='val')
    test_data = CellsDataset(dataset_fname, which_set='test')

    # create dataloaders
    # num_workers=4 crashes in my spyder but works on pure python
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        test_data, shuffle=True, batch_size=batch_size, num_workers=4)

    # test loading I guess
    for tc, (imgs, labs) in enumerate(train_loader):
        print(imgs.shape, labs.shape)

    print(f'{tc} iterations')













