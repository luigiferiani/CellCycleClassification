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
from torch.utils.data.sampler import WeightedRandomSampler


class CellsDatasetBase(Dataset):

    LABELS_COL = []
    INFO_COLS = [
        'frame',
        'x_center',
        'y_center',
        'track_id',
        'unique_track_id',
        'ind_in_annotations_df',
        'curated_label_id',
        'label_is_S_phase']

    def __init__(
            self,
            hdf5_filename,
            which_set='train',
            roi_size=80,
            labels_dtype=torch.long,
            is_use_default_transforms=True):

        self.fname = hdf5_filename
        self.set_name = which_set + '_df'
        self.label_info = []
        self._is_return_extra_info = False
        self._all_dfs = None
        # # get labels info
        # ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        # self.label_info = ann_df[self.INFO_COLS].copy()
        self.roi_size = roi_size  # size we want to train on
        self.labels_dtype = labels_dtype

        self.frame_height, self.frame_width = (
            self._get_frame(0).squeeze().shape[-2:])

        # any transform?
        self.default_transforms_list = [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomErasing(),
                ]
        if is_use_default_transforms and which_set in ['train', 'val']:
            self.transform = transforms.Compose(self.default_transforms_list)
        else:
            self.transform = transforms.Compose([])  # does nothing

    @property
    def is_return_extra_info(self):
        return self._is_return_extra_info

    @is_return_extra_info.setter
    def is_return_extra_info(self, new_value):
        assert isinstance(new_value, bool)
        if (
                (not self._is_return_extra_info) and
                (new_value is True) and
                (self._all_dfs is None)
                ):
            self._all_dfs = pd.concat(
                [pd.read_hdf(self.fname, dfname)
                 for dfname in ['/train_df', '/val_df', '/test_df']],
                axis=0,
                ignore_index=True)
        self._is_return_extra_info = new_value

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
        labels = label_info[self.LABELS_COL]
        labels = np.array(labels)
        labels = torch.from_numpy(labels).type(self.labels_dtype)

        is_first_in_stage = None
        track_id = None
        frame_number = None
        if self.is_return_extra_info:
            is_first_in_stage = self.is_first_frame_of_stage(label_info)
            track_id = label_info['unique_track_id']
            frame_number = label_info['frame']

        return img, labels, is_first_in_stage, track_id, frame_number

    # internal function to get a ROI's pixel data
    def _get_roi(self, info):

        frame_number = int(info['frame'])  # for some reason it was a float...

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
        roi_data = self._get_frame(frame_number)[
            min_row:max_row, min_col:max_col]

        # pad with constant value, if necessary
        if np.any((pad_v, pad_h)):
            roi_data = np.pad(
                roi_data, (pad_v, pad_h),
                mode='constant',
                constant_values=np.percentile(roi_data, 5))

        assert roi_data.shape == (self.roi_size, self.roi_size)

        return roi_data

    def _get_frame(self, frame_number):
        with tables.File(self.fname, 'r') as fid:
            frame_data = fid.get_node('/full_data')[frame_number].copy()
        return frame_data

    def img_normalise(self, img):
        img -= img.mean()
        img /= img.std()
        return img

    def set_use_transforms(self, is_use_transforms):
        if is_use_transforms:
            self.transform = transforms.Compose(self.default_transforms_list)
        else:
            self.transform = transforms.Compose([])  # does nothing
        return

    def is_first_frame_of_stage(self, this_roi):
        """Return True if this cell, in the previous frame,
        had a different label
        Assume that original dataframe was sorted by track_id then by frame"""

        is_first_in_stage = False
        # if it's the first roi in the video, we don't know the previous stage
        if this_roi['ind_in_annotations_df'] != 0:
            # previous roi needs to exist,
            # with same track id but different label
            prev_roi = self._find_previous_roi(this_roi)
            if prev_roi is not None:
                this_id = this_roi['track_id']
                this_label = this_roi['curated_label_id']
                prev_id = prev_roi['track_id']
                prev_label = prev_roi['curated_label_id']
                if (prev_id == this_id) and (prev_label != this_label):
                    is_first_in_stage = True
        return is_first_in_stage

    def _find_previous_roi(self, this_roi):
        iiad = 'ind_in_annotations_df'
        if this_roi[iiad] == 0:
            return None
        idx = self._all_dfs[iiad] == (this_roi[iiad] - 1)
        assert idx.sum() <= 1, 'previous roi found multiple times?'
        if idx.sum() == 0:
            return None

        prev_roi = self._all_dfs[idx].iloc[0]
        return prev_roi


class CellsDataset(CellsDatasetBase):

    LABELS_COL = 'label_is_S_phase'

    def __init__(
            self,
            hdf5_filename,
            which_set='train',
            roi_size=80,
            labels_dtype=torch.long,
            is_use_default_transforms=True):

        super().__init__(
            hdf5_filename,
            which_set=which_set,
            roi_size=roi_size,
            labels_dtype=labels_dtype,
            is_use_default_transforms=is_use_default_transforms,
            )

        ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        self.label_info = ann_df[self.INFO_COLS].copy()


class CellsDatasetMultiClass(CellsDatasetBase):

    LABELS_COL = 'multiclass_label_id'

    def __init__(
            self,
            hdf5_filename,
            which_set='train',
            roi_size=80,
            labels_dtype=torch.long,
            is_use_default_transforms=True):

        super().__init__(
            hdf5_filename,
            which_set=which_set,
            roi_size=roi_size,
            labels_dtype=labels_dtype,
            is_use_default_transforms=is_use_default_transforms,
            )

        # get labels info
        ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        self.label_info = ann_df[self.INFO_COLS].copy()
        # class label needs be 0:N-1 for loss fn
        self.label_info[self.LABELS_COL] = -1 + self.label_info[
            'curated_label_id'].values

        # look at classes imbalance
        labels_counts = self.label_info[self.LABELS_COL].value_counts()
        labels_weights = 1./labels_counts
        self.samples_weights = labels_weights[
            self.label_info[self.LABELS_COL]].values
        # this is now an array with higher values for worse-represented classes


class CellsDatasetMultiClassNew(CellsDatasetBase):

    LABELS_COL = 'multiclass_label_id'
    LABEL_MERGE_DICT = {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 3}
    LABEL_DICT = {
        0: 'G0/1',
        1: 'S',
        2: 'G2',
        3: 'M'
        }

    def __init__(
            self,
            hdf5_filename,
            which_set='train',
            roi_size=80,
            labels_dtype=torch.long,
            is_use_default_transforms=True):

        super().__init__(
            hdf5_filename,
            which_set=which_set,
            roi_size=roi_size,
            labels_dtype=labels_dtype,
            is_use_default_transforms=is_use_default_transforms,
            )

        # get labels info
        ann_df = pd.read_hdf(hdf5_filename, key='/'+self.set_name)
        self.label_info = ann_df[self.INFO_COLS].copy()
        # class label needs be 0:N-1 for loss fn
        self.label_info[self.LABELS_COL] = self.label_info[
            'curated_label_id'].map(self.LABEL_MERGE_DICT)

        # look at classes imbalance
        labels_counts = self.label_info[self.LABELS_COL].value_counts()
        labels_weights = 1./labels_counts
        self.samples_weights = labels_weights[
            self.label_info[self.LABELS_COL]].values
        # this is now an array with higher values for worse-represented classes



# %%

if __name__ == "__main__":

    from pathlib import Path

    # where are things?
    work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    work_dir /= 'new_annotated_datasets'
    # dataset_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    dataset_fname = work_dir / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'

    # parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 64

    # create datasets
    train_data = CellsDataset(dataset_fname, which_set='train', roi_size=48)
    val_data = CellsDataset(dataset_fname, which_set='val', roi_size=48)
    test_data = CellsDataset(dataset_fname, which_set='test', roi_size=48)

    # train_data = CellsDatasetMultiClass(dataset_fname, which_set='train', roi_size=48)
    # val_data = CellsDatasetMultiClass(dataset_fname, which_set='val', roi_size=48)
    # test_data = CellsDatasetMultiClass(dataset_fname, which_set='test', roi_size=48)

    # def _get_sampler(dtst):
    #     return WeightedRandomSampler(
    #         dtst.samples_weights, len(dtst.samples_weights)*2, replacement=True)

    # train_sampler = _get_sampler(train_data)
    # val_sampler = _get_sampler(val_data)
    # test_sampler = _get_sampler(test_data)


    # create dataloaders
    # num_workers=4 crashes in my spyder but works on pure python
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        test_data, shuffle=True, batch_size=batch_size, num_workers=4)

    # train_loader = DataLoader(
    #     train_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)
    # val_loader = DataLoader(
    #     val_data, sampler=val_sampler, batch_size=batch_size, num_workers=4)
    # test_loader = DataLoader(
    #     test_data, sampler=test_sampler, batch_size=batch_size, num_workers=4)

    # test loading I guess
    labsacc = []
    for tc, (imgs, labs) in enumerate(train_loader):
        print(imgs.shape, labs.shape)
        labsacc.append(labs)

    for tc, (testimgs, testlabs) in enumerate(test_loader):
        print(testimgs.shape)

    print(f'{tc} iterations')


    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axs[0].imshow(testimgs[0].squeeze())
    axs[0].set_title(testlabs[0])
    axs[1].imshow(imgs[1].squeeze())
    # %%
    for data in [val_data, test_data]:
        data.is_return_extra_info = True
        foo = [(label.item(), isfirst) for (_, label, isfirst) in data]
        bar = pd.DataFrame(foo, columns=['label_id', 'is_first'])
        print(bar.value_counts())

    # transfunc = transforms.Compose([
    #     transforms.ToPILImage(mode='F')])
    # fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # axs[0].imshow(testimgs[0].squeeze())
    # axs[1].imshow(transfunc(testimgs[0]))

    # %%
    this_roi_info = val_data.label_info.loc[1287]
    ref_ind = this_roi_info['ind_in_annotations_df']
    foo = val_data._all_dfs.query(
        f'(ind_in_annotations_df > {ref_ind}-30) and (ind_in_annotations_df < {ref_ind}+30)')
    foo = foo.sort_values(by='ind_in_annotations_df')












