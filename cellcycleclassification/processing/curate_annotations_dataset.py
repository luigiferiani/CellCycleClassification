#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:18:44 2020

@author: lferiani

Annotations are only labelled in the first frame of each cell cycle stage
For training, all ROIS need to be annotated at all frames

DEPRECATED - USE MAKE_DATASET_FOR_DL INSTEAD

"""

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

LABELS_DICT = {
    1: 'g0',
    2: 'g1',
    3: 's',
    4: 'g2',
    5: 'm',
    6: 'nocell',
    7: 'unsure'}

BAD_LABELS = [6, 7]


def plot_labels(df):
    fig, ax = plt.subplots()
    for t_id, t_df in df.groupby('track_id'):
        t_df.plot(y='curated_label_id', ax=ax)
    ax.get_legend().remove()


# %%

# where are things
work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
annotations_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
ann_df = pd.read_hdf(annotations_fname, key='/annotations_df')


# copy the original annotations column
ann_df['curated_label_id'] = ann_df['label_id'].copy()
# write nans instead of zeros
idx_zeros = ann_df['curated_label_id'] == 0
ann_df.loc[idx_zeros, 'curated_label_id'] = np.nan
# fill nans forwards. This propagates the "first frame of state" label
ann_df['curated_label_id'] = ann_df.groupby(
    'track_id')['curated_label_id'].ffill()
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
    # only case it decreases is M -> G0
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
            np.any(t_df.loc[prev_ind, 'curated_label_id'] == BAD_LABELS)
            )
        if not acceptable:
            bad_track_id.append(t_id)

# now filter dataset:
# exclude the bad tracks
clean_df = ann_df[~ann_df['track_id'].isin(bad_track_id)].copy()
clean_df = clean_df[~clean_df['curated_label_id'].isin(BAD_LABELS)]


plt.close('all')
plot_labels(ann_df)
plot_labels(clean_df)

# create a binary label that we're gonna use for training
clean_df['label_is_S_phase'] = clean_df['curated_label_id'] == 3

print(clean_df['label_is_S_phase'].value_counts())

# %% now split curated dataset
# get sizes
n_rois = clean_df.shape[0]
test_frac = 0.2
val_frac = 0.2
test_size = int(test_frac * n_rois)
val_size = int(val_frac * n_rois)
train_size = n_rois - test_size - val_size


# get an index *to use with iloc*
ind = np.arange(n_rois, dtype=int)
np.random.shuffle(ind)

# split it in 3 sets
split_ind = {'test': ind[:test_size],
             'val': ind[test_size:test_size+val_size],
             'train': ind[test_size+val_size:]}
assert split_ind['train'].shape[0] == train_size

# loop on sets
for set_name, ilocind in split_ind.items():
    # get portion of df, drop old index
    set_df = clean_df.iloc[ilocind]
    set_df = set_df.rename_axis('ind_in_annotations_df').reset_index()
    # write to disk
    set_df.to_hdf(annotations_fname,
                  key='/'+set_name+'_df')

