#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:02:12 2020

@author: lferiani

in the annotated hdf5 on onedrive, i cannot read all frames in full_data.
this code takes the full_data and from an uncorrputed source, and updates
annotations_df only
"""

import shutil
import pandas as pd

from pathlib import Path

work_dir = Path('/Users/lferiani/work_repos/CellCycleClassification/data/')
good_fname = work_dir / 'R5C5F1_PCNA_sel_annotations_todo.hdf5'
corr_fname = work_dir / 'R5C5F1_PCNA_sel_annotations_done.hdf5'
dst_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'

# create destination file first by copying the dataset with the good images
shutil.copy(good_fname, dst_fname)
annotations_df = pd.read_hdf(corr_fname, '/annotations_df')
annotations_df.to_hdf(dst_fname, '/annotations_df')