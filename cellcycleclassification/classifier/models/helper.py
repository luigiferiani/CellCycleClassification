#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:10:28 2020

@author: lferiani
"""

import torch.nn
from cellcycleclassification.classifier import datasets
from cellcycleclassification.classifier.models import cnn_tierpsy


AVAILABLE_MODELS = {
    'cnn_tierpsy': cnn_tierpsy.CNN_tierpsy(),
    'cnn_tierpsy_roi48': cnn_tierpsy.CNN_tierpsy_roi48(),
    'cnn_tierpsy_roi48_v2': cnn_tierpsy.CNN_tierpsy_roi48_v2(),
    'cnn_tierpsy_roi48_v3': cnn_tierpsy.CNN_tierpsy_roi48_v3(),
    'cnn_tierpsy_roi48_v4': cnn_tierpsy.CNN_tierpsy_roi48_v4(),
    }


def get_dataset(model_name, which_split, data_path):
    """return the correct dataset given a model name"""
    if model_name == 'cnn_tierpsy':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=80)

    elif model_name == 'cnn_tierpsy_roi48':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48)

    elif model_name == 'cnn_tierpsy_roi48_v2':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48)

    elif model_name == 'cnn_tierpsy_roi48_v3':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48)

    elif model_name == 'cnn_tierpsy_roi48_v4':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48)

    else:
        raise ValueError('case not coded yet')

    return dataset


def get_loss_criterion(model_name):
    if model_name in [
            'cnn_tierpsy',
            'cnn_tierpsy_roi48',
            'cnn_tierpsy_roi48_v2',
            'cnn_tierpsy_roi48_v3',
            ]:
        criterion = torch.nn.CrossEntropyLoss()
    if model_name in [
            'cnn_tierpsy_roi48_v4',
            ]:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('case not coded yet')

    return criterion


def get_model_datasets_criterion(model_name, which_splits=[], data_path=None):
    """
    Look up the model name in a dict of available models.
    Match it to the appropriate data loader
    """

    if len(which_splits) > 0:
        assert data_path is not None, 'Path to data missing'

    if isinstance(which_splits, str):
        which_splits = [which_splits]

    # grab correct model
    if model_name in AVAILABLE_MODELS.keys():
        model_instance = AVAILABLE_MODELS[model_name]
    else:
        errstr = (f'{model_name} not in the list of available models: '
                  + f'{list(AVAILABLE_MODELS.keys())}')
        raise ValueError(errstr)

    # grab correct loss criterion
    criterion = get_loss_criterion(model_name)

    # grab corect dataset
    datasets = tuple(
        get_dataset(model_name, split, data_path)
        for split in which_splits)

    # if input for which_splits was a single split name,
    # return a single dataset not in a tuple
    if len(datasets) == 1:
        datasets = datasets[0]

    return model_instance, criterion, datasets
