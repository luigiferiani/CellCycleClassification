#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:10:28 2020

@author: lferiani
"""

import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from cellcycleclassification.classifier import datasets
from cellcycleclassification.classifier.models import cnn_tierpsy


AVAILABLE_MODELS = {
    'cnn_tierpsy': cnn_tierpsy.CNN_tierpsy(),
    'cnn_tierpsy_roi48': cnn_tierpsy.CNN_tierpsy_roi48(),
    'cnn_tierpsy_roi48_v2': cnn_tierpsy.CNN_tierpsy_roi48_v2(),
    'cnn_tierpsy_roi48_v3': cnn_tierpsy.CNN_tierpsy_roi48_v3(),
    'cnn_tierpsy_roi48_v4': cnn_tierpsy.CNN_tierpsy_roi48_v4(),
    'cnn_tierpsy_roi48_multi': cnn_tierpsy.CNN_tierpsy_roi48_multiclass(),
    'cnn_tierpsy_roi48_original': cnn_tierpsy.CNN_tierpsy_roi48_original(),
    'cnn_tierpsy_roi48_original_v2':
        cnn_tierpsy.CNN_tierpsy_roi48_original_v2(),
    'cnn_tierpsy_roi48_original_v3':
        cnn_tierpsy.CNN_tierpsy_roi48_original_v3(),
    'cnn_tierpsy_roi48_original_v4':
        cnn_tierpsy.CNN_tierpsy_roi48_original_v4(),
    'cnn_tierpsy_roi48_original_v5':
        cnn_tierpsy.CNN_tierpsy_roi48_original_v5(),
    'cnn_tierpsy_roi48_original_multi':
        cnn_tierpsy.CNN_tierpsy_roi48_original_multiclass(),
    'cnn_tierpsy_roi48_original_multi_v2':
        cnn_tierpsy.CNN_tierpsy_roi48_original_multiclass_v2(),
    'cnn_tierpsy_roi48_original_multi_v3':
        cnn_tierpsy.CNN_tierpsy_roi48_original_multiclass_v3(),
    'cnn_tierpsy_roi48_original_multi_v4':
        cnn_tierpsy.CNN_tierpsy_roi48_original_multiclass_v4(),
    'cnn_tierpsy_roi48_original_multi_v5':
        cnn_tierpsy.CNN_tierpsy_roi48_original_multiclass_v5()
    }


def get_dataset(model_name, which_split, data_path):
    """return the correct dataset given a model name"""
    if model_name == 'cnn_tierpsy':
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=80)

    elif model_name in [
            'cnn_tierpsy_roi48',
            'cnn_tierpsy_roi48_v2',
            'cnn_tierpsy_roi48_v3',
            ]:
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48)

    elif model_name in [
            'cnn_tierpsy_roi48_v4',
            'cnn_tierpsy_roi48_original',
            'cnn_tierpsy_roi48_original_v2',
            'cnn_tierpsy_roi48_original_v3',
            'cnn_tierpsy_roi48_original_v4',
            'cnn_tierpsy_roi48_original_v5',
            ]:
        dataset = datasets.CellsDataset(
            data_path, which_set=which_split, roi_size=48,
            labels_dtype=torch.float)

    elif model_name == 'cnn_tierpsy_roi48_multi':
        dataset = datasets.CellsDatasetMultiClass(
            data_path, which_set=which_split, roi_size=48,
            labels_dtype=torch.long)

    elif model_name in [
            'cnn_tierpsy_roi48_original_multi',
            'cnn_tierpsy_roi48_original_multi_v2',
            'cnn_tierpsy_roi48_original_multi_v3',
            'cnn_tierpsy_roi48_original_multi_v4',
            'cnn_tierpsy_roi48_original_multi_v5',
            ]:
        dataset = datasets.CellsDatasetMultiClassNew(
            data_path, which_set=which_split, roi_size=48,
            labels_dtype=torch.long)
    else:
        raise ValueError('case not coded yet')

    return dataset


def get_loss_criterion(model_name):
    if model_name in [
            'cnn_tierpsy',
            'cnn_tierpsy_roi48',
            'cnn_tierpsy_roi48_v2',
            'cnn_tierpsy_roi48_v3',
            'cnn_tierpsy_roi48_multi',
            'cnn_tierpsy_roi48_original_multi',
            'cnn_tierpsy_roi48_original_multi_v2',
            'cnn_tierpsy_roi48_original_multi_v3',
            'cnn_tierpsy_roi48_original_multi_v4',
            'cnn_tierpsy_roi48_original_multi_v5',
            ]:
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name in [
            'cnn_tierpsy_roi48_v4',
            'cnn_tierpsy_roi48_original',
            'cnn_tierpsy_roi48_original_v2',
            'cnn_tierpsy_roi48_original_v3',
            'cnn_tierpsy_roi48_original_v4',
            'cnn_tierpsy_roi48_original_v5',
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


def get_dataloader(dataset, is_use_sampler, batch_size, num_workers):
    if is_use_sampler:
        sampler = WeightedRandomSampler(
            dataset.samples_weights, len(dataset), replacement=True)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers)
    else:
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers)
    return loader

