#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:58:55 2020

@author: lferiani

Annotations are only labelled in the first frame of each cell cycle stage
For training, all ROIS need to be annotated at all frames

This deals with having multiple annotated files

"""

import tables
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)

LABELS_DICT = {
    1: 'g0',
    2: 'g1',
    3: 's',
    4: 'g2',
    5: 'm',
    6: 'nocell',
    7: 'unsure'}

BAD_LABELS = [6, 7]


def plot_labels(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for t_id, t_df in df.groupby('track_id'):
        t_df.plot(y='curated_label_id', ax=ax)
    ax.get_legend().remove()


def curate_annotations_df(annotations_fname, video_id):
    """
    - load annotations
    - propagate annotations to frames after first of a stage
    - check for wrong annotations (eg a G1 after S)
    - drop uncertain and no cell
    - also create binary label for S/not S binary classifier
    """

    # load annotations
    ann_df = pd.read_hdf(annotations_fname, key='/annotations_df')
    # copy the original annotations column
    ann_df['curated_label_id'] = ann_df['label_id'].copy()
    # write nans instead of zeros
    idx_zeros = ann_df['curated_label_id'] == 0
    ann_df.loc[idx_zeros, 'curated_label_id'] = np.nan
    # fill nans forwards. This propagates the "first frame of state" label
    ann_df['curated_label_id'] = ann_df.groupby(
        'track_id')['curated_label_id'].ffill()
    # assuming it's ok to bfill too
    # after manually checking, a few (5) rois were still nans. it's ok to bfill
    ann_df['curated_label_id'] = ann_df.groupby(
        'track_id')['curated_label_id'].bfill()
    ann_df['curated_label_id'] = ann_df['curated_label_id'].astype(int)

    # checks, add bad tracks to list
    bad_track_id = []
    for t_id, t_df in ann_df.groupby('track_id'):
        # labels diff
        dd = t_df['curated_label_id'].diff()
        # label should always increase.
        # only case it decreases is M -> G0 (or m->G1 when Alexis annotated)
        # or uncertain -> any label
        # or nocell -> any label
        idx_neg_dd = dd < 0
        ind_neg_dd = idx_neg_dd[idx_neg_dd].index.to_list()
        ind_prev_dd = [ind-1 for ind in ind_neg_dd]
        neg_dd_couples = list(zip(ind_prev_dd, ind_neg_dd))
        for prev_ind, neg_ind in neg_dd_couples:
            acceptable = (
                t_df.loc[prev_ind, 'curated_label_id'] == 5 and
                t_df.loc[neg_ind, 'curated_label_id'] == 1)
            acceptable = acceptable or (
                t_df.loc[prev_ind, 'curated_label_id'] == 5 and
                t_df.loc[neg_ind, 'curated_label_id'] == 2)
            acceptable = acceptable or (
                np.any(t_df.loc[prev_ind, 'curated_label_id'] == BAD_LABELS)
                )
            if not acceptable:
                bad_track_id.append(t_id)

    # now filter dataset:
    # exclude the bad tracks
    clean_df = ann_df[~ann_df['track_id'].isin(bad_track_id)].copy()
    clean_df = clean_df[~clean_df['curated_label_id'].isin(BAD_LABELS)]

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    plot_labels(ann_df, ax=axs[0])
    plot_labels(clean_df, ax=axs[1])

    # create a binary label that we're gonna use for training
    clean_df['label_is_S_phase'] = clean_df['curated_label_id'] == 3

    print(clean_df['label_is_S_phase'].value_counts())
    print(clean_df.shape)

    # also add the video id. Acts in place
    clean_df.insert(0, 'video_id', video_id)

    return clean_df


# %% where are things
# inputs
work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
work_dir /= 'new_annotated_datasets'
annotations_fnames = [
    'R5C5F1_PCNA_sel_annotations_done_luigi.hdf5',
    'R5C5F2_PCNA_sel_annotations.hdf5',
    'R5C5F3_PCNA_sel_annotations.hdf5',
    ]
# make absolute
annotations_fnames = [work_dir / af for af in annotations_fnames]

# output
output_hdf5 = work_dir / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'

# %% load data
# ann_set_id is the same as video_id in curate_annotations_df:
# video data from dataset 0 will be at index 0 along the first dimension
# of full_data (n_datasets-by-frames-by-height-by-width)
clean_dfs = []
for ann_set_id, annotations_fname in enumerate(annotations_fnames):
    this_clean_df = curate_annotations_df(annotations_fname, ann_set_id)
    clean_dfs.append(this_clean_df)
big_clean_df = pd.concat(clean_dfs, axis=0, sort=False).reset_index(drop=True)

# how many tracks do we have?
big_clean_df.insert(
    2,
    'unique_track_id',
    big_clean_df.groupby(by=['video_id', 'track_id']).ngroup()
    )

# fix frame number: full_data will be concatenated
# store old frame number
big_clean_df = big_clean_df.rename(columns={'frame': 'frame_in_vid'})
# get videos sizes
vid_sizes = np.zeros(len(annotations_fnames))
for ann_set_id, annotations_fname in enumerate(annotations_fnames):
    with tables.File(annotations_fname, 'r') as fid:
        vid_sizes[ann_set_id] = fid.get_node('/full_data').shape[0]
# transform them in offsets
frame_offset = np.cumsum(vid_sizes) - vid_sizes[0]
# and add them to the frames
big_clean_df['frame'] = (
    frame_offset[big_clean_df['video_id']] + big_clean_df['frame_in_vid']
    )


# %% now split curated dataset
# as of 27/10/2020, let's keep annotations 3 as the test set
# (it accounts for about 15% of the unique tracks)
# the rest will be split in train and validation according to unique_track_id

# tracks to split
tracks_to_split = big_clean_df.query('video_id != 2')[
    'unique_track_id'].unique()

# use train_test_split to get train/val split
tracks_train, tracks_val = train_test_split(
    tracks_to_split, test_size=0.2, random_state=20201027)
# for completeness, let's manualy create the tracks_test array
tracks_test = big_clean_df.query('video_id == 2')['unique_track_id'].unique()


# %% get sizes for info only

n_tracks = big_clean_df['unique_track_id'].nunique()

n_train_tracks = tracks_train.shape[0]
n_val_tracks = tracks_val.shape[0]
n_test_tracks = tracks_test.shape[0]

n_train_rois = big_clean_df['unique_track_id'].isin(tracks_train).sum()
n_val_rois = big_clean_df['unique_track_id'].isin(tracks_val).sum()
n_test_rois = big_clean_df['unique_track_id'].isin(tracks_test).sum()

print((
       'Tracks split:\n'
       f'{n_train_tracks}\ttrain\n'
       f'{n_val_tracks}\tval\n'
       f'{n_test_tracks}\ttest\n')
      )

print((
       'ROIs split:\n'
       f'{n_train_rois}\ttrain\n'
       f'{n_val_rois}\tval\n'
       f'{n_test_rois}\ttest\n')
      )

# %% write datasets in training dataset, shuffling train and val but not test


def _save_set(set_unique_tracks, set_name, is_shuffle=False):
    # get filtering
    idx_bool_tosave = big_clean_df['unique_track_id'].isin(set_unique_tracks)
    # apply filtering
    set_df = big_clean_df[idx_bool_tosave].copy(deep=True)
    if is_shuffle:
        # shuffle (random selection with no sub of the whole sample == shuffle)
        set_df = set_df.sample(frac=1)
    # and reset index while keeping the old one
    set_df = set_df.rename_axis('ind_in_annotations_df').reset_index()
    # write to disk
    set_df.to_hdf(output_hdf5,
                  key='/'+set_name+'_df')
    return


# save things on disk
# first the big annotations set (and delete whatever was in that file)
big_clean_df.to_hdf(output_hdf5, '/annotations_df', mode='w')
# then the sub-sets
_save_set(tracks_train, 'train', is_shuffle=True)
_save_set(tracks_val, 'val', is_shuffle=True)
_save_set(tracks_test, 'test', is_shuffle=False)


# %% full data

# read them all
# memory will be fine if I do it the naive way
all_full_data = []
for ann_set_id, annotations_fname in enumerate(annotations_fnames):
    with tables.File(annotations_fname, 'r') as fid:
        all_full_data.append(
            fid.get_node('/full_data').read()
            )
all_full_data = np.concatenate(all_full_data, axis=0)

# now write
with tables.File(output_hdf5, 'a') as fid:
    fid.create_earray(
        '/',
        'full_data',
        obj=all_full_data,
        filters=FILTERS)





