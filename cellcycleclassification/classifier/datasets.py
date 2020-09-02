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

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CellsDataset(Dataset):

    LABELS_COLS = ['frame', 'x_center', 'y_center', 'label_is_S_phase']

    def __init__(self, hdf5_filename, which_set='train', roi_size=80):

        self.fname = hdf5_filename
        self.set_name = which_set + '_df'
        # get labels info
        ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        self.label_info = ann_df[self.LABELS_COLS]
        self.roi_size = roi_size  #80  # size we want to train on
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
        # read images from disk, cast to single
        roi_data = self._get_roi(label_info).astype(np.float32)

        # normalize and fix dimensions for pytorch
        img = self.img_normalise(roi_data)[None, :, :]
        # make a tensor
        img = torch.from_numpy(img)
        # apply transformation
        img = self.transform(img)

        # read labels too
        labels = label_info['label_is_S_phase']
        # labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
        labels = np.array(labels)
        labels = torch.from_numpy(labels).long()

        return img, labels

    # internal function to get a ROI's pixel data
    def _get_roi(self, info):

        frame_number = info['frame']

        def fix_extrema(centre, half_roi_size, max_of_dim):
            """
            cap minimum and maximum of roi boundaries to fit in frame,
            also return padding to add to keep fixed size
            """
            min_ = centre - half_roi_size
            max_ = centre + half_roi_size
            pad_neg = -min(0, min_)
            min_ = max(0, min_)
            pad_pos = max(0, max_-max_of_dim)
            max_ = min(max_, max_of_dim)
            return min_, max_, (pad_neg, pad_pos)

        min_col, max_col, pad_h = fix_extrema(
            int(info['x_center']), self.roi_size // 2, self.frame_width)
        min_row, max_row, pad_v = fix_extrema(
            int(info['y_center']), self.roi_size // 2, self.frame_height)

        # get roi data
        with tables.File(self.fname, 'r') as fid:
            roi_data = fid.get_node('/full_data')[
                frame_number, min_row:max_row, min_col:max_col].copy()

        # pad with constant zeros, if necessary
        if np.any((pad_v, pad_h)):
            # roi_data = np.pad(roi_data, ((pad_t, pad_b), (pad_l, pad_r)))
            roi_data = np.pad(roi_data, (pad_v, pad_h))

        assert roi_data.shape == (self.roi_size, self.roi_size)

        return roi_data

    def img_normalise(self, img):
        img -= img.mean()
        img /= img.std()
        return img


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













