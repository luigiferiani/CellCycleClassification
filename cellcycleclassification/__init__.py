#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:56:02 2021

@author: lferiani
"""
from pathlib import Path

base_path = Path(__file__).parent
BINARY_MODEL_PATH = base_path / 'trained_models' / 'v_06_60_best.pth'
MULTICLASS_MODEL_PATH = (
    base_path / 'trained_models' / 'v_12_63_20201218_213041.pth')
