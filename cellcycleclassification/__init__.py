#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:56:02 2021

@author: lferiani
"""
from pathlib import Path

base_path = Path(__file__).parent
trained_models_path = base_path / 'trained_models'
BINARY_MODEL_PATH = trained_models_path / 'v_06_60_20210802_192216.pth'
MULTICLASS_MODEL_PATH = trained_models_path / 'v_12_63_20210802_192250.pth'

repo_path = base_path.parent
MANUAL_ANNOTATIONS_PATH = repo_path / 'data' / 'manual_annotations'
DL_DATASET_PATH = MANUAL_ANNOTATIONS_PATH / 'DL_datasets'