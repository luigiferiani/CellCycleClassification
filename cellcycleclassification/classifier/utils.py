#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:30:33 2020

@author: lferiani
"""

import torch
import warnings
import platform
from pathlib import Path


POSSIBLE_DATA_PATHNAMES = [
    (Path.home()
     / 'work_repos/CellCycleClassification/data/new_annotated_datasets'
     # / 'R5C5F1_PCNA_sel_annotations.hdf5'),
     # / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'),
     / 'R5C5F_PCNA_dl_dataset_20201216.hdf5'),
    (Path.home()
     / 'work_repos/CellCycleClassification/data/'
     # / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'),
     / 'R5C5F_PCNA_dl_dataset_20201216.hdf5'),
    ]


def get_default_data_path():
    out = None
    for candidate in POSSIBLE_DATA_PATHNAMES:
        if candidate.exists():
            out = candidate
            break
    if out is None:
        raise Exception('Cannot find the dataset in the default paths')
    return out


def get_default_log_dir():
    platname = platform.system()

    bg_mac = Path('/Volumes/behavgenom$/')
    bg_linux = Path.home() / 'net/behavgenom$/'

    log_on_bg = Path(
        'Luigi/Data/AlexisBarr_cell_cycle_classification/trained_models')

    if platname == 'Darwin':
        logdir = bg_mac / log_on_bg
    elif platname == 'Linux':
        logdir = bg_linux / log_on_bg
    else:
        raise Exception('not coded for windows yet')

    if not logdir.exists():
        from cellcycleclassification import base_path
        local_logdir = base_path / 'trained_models'

        logdir = local_logdir
        warnings.warn(
            f'remote logdir not found, logging instead in {logdir}'
            )

    return logdir


def set_device(cuda_id):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            device = torch.device(f"cuda:{cuda_id}")
        except:
            device = torch.device("cuda")
            warnings.warn("could not use cuda:{cuda_id}")
    else:
        device = torch.device('cpu')

    print(f'Device in use: {device}')
    if use_cuda:
        print(torch.cuda.get_device_name())

    return device


def sanitise_path(_input, _which):

    if _input is None:
        if _which == 'dataset':
            get_default_path = get_default_data_path
        elif _which == 'log_dir':
            get_default_path = get_default_log_dir
        else:
            raise ValueError(f'{_which} case not coded yet')
        _input = get_default_path()

    elif isinstance(_input, str):
        _input = Path(_input)

    return _input


def get_training_parameters(session_name):
    from cellcycleclassification.classifier.scripts.train import SESSIONS
    try:
        pars_dict = SESSIONS[session_name]
    except KeyError:
        print(f'cannot find parameters for {session_name}')
        return
    # patch parameters that were added later
    if 'is_use_sampler' not in pars_dict.keys():
        pars_dict['is_use_sampler'] = False
    if 'roi_size' not in pars_dict.keys():
        pars_dict['roi_size'] = None
    return pars_dict