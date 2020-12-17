#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:57:50 2020

@author: lferiani

A series of single frames with high content of mitotic nuclei were annotated
using a different GUI. In this annotated set, the only important variable was
the location of M nuclei.

This script convert that annotation (hdf5 + csv) into an hdf5 dataset like
the ones created by crete_annotations_dataset.py + the CellCycleAnnotator GUI
"""

import shutil
import pandas as pd
from pathlib import Path

COLS_DICT = {
    'frame_number': 'frame',
    'x': 'x_center',
    'y': 'y_center',
    }

INT_COLS = ['track_id', 'frame', 'label_id']

# where are things?
src_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
dst_dir = src_dir / 'new_annotated_datasets'

dataset_stem = 'Mitotic_A549_nuclei'
src_hdf5_fname = src_dir / (dataset_stem + '.hdf5')
dst_hdf5_fname = dst_dir / (dataset_stem + '.hdf5')

src_anns_fname = src_dir / (dataset_stem + '_eggs.csv')

# first, copy the hdf5 to its new location, so we do not corrupt the original
shutil.copy2(src_hdf5_fname, dst_hdf5_fname)

# then we open the csv
anns_df = pd.read_csv(src_anns_fname)

# let's create the columns that will be needed downstream
# track_id: each annotation is a different cell so it's just unique numbers
anns_df['track_id'] = anns_df.index.to_list()
# label_id: everything here is Mitosis. so annotation label is 5
anns_df['label_id'] = 5
# nothing was interpolated so
anns_df['is_interpolated'] = False
# rename columns
anns_df = anns_df.rename(columns=COLS_DICT)
# remove the group_name column
anns_df = anns_df.drop(columns=['group_name'])
# force type
for col in INT_COLS:
    anns_df[col] = anns_df[col].astype(int)

# that's it. now let's write in the same dataset that has the /full_data
anns_df.to_hdf(
    dst_hdf5_fname,
    key='/annotations_df',
    index=False,
    mode='r+')

