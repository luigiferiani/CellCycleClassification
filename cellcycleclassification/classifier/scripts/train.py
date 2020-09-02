#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:34:45 2020

@author: lferiani
"""

from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cellcycleclassification.classifier.utils import set_device, sanitise_path
from cellcycleclassification.classifier.trainer.engine import train_model
from cellcycleclassification.classifier.models.helper import (
    get_model_datasets_criterion)


SESSIONS = dict(
    v_debug=dict(
        model_name='cnn_tierpsy',
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=5,
        num_workers=4,
        scheduler=None,
        ),
    v_00_00=dict(
        model_name='cnn_tierpsy',
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=120,
        num_workers=4,
        scheduler=None,
        ),  # done
    v_00_01=dict(
        model_name='cnn_tierpsy',
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=200,
        num_workers=4,
        scheduler=None,
        ),
    v_00_02=dict(
        model_name='cnn_tierpsy',
        batch_size=32,
        learning_rate=1e-4,
        n_epochs=200,
        num_workers=4,
        scheduler=None,
        ),
    v_00_03=dict(
        model_name='cnn_tierpsy',
        batch_size=128,
        learning_rate=3e-4,
        n_epochs=200,
        num_workers=4,
        scheduler=None,
        ),
    v_00_04=dict(
        model_name='cnn_tierpsy',
        batch_size=128,
        learning_rate=1e-4,
        n_epochs=200,
        num_workers=4,
        scheduler=None,
        ),
    v_00_05=dict(
        model_name='cnn_tierpsy',
        batch_size=32,
        learning_rate=3e-4,
        n_epochs=200,
        num_workers=4,
        scheduler=None,
        ),
    v_00_50=dict(
        model_name='cnn_tierpsy',
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=200,
        num_workers=4,
        scheduler=ReduceLROnPlateau,
        scheduler_kwargs=dict(
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
            ),
        ),
    v_00_51=dict(
        model_name='cnn_tierpsy',
        batch_size=32,
        learning_rate=1e-3,
        n_epochs=200,
        num_workers=4,
        scheduler=ReduceLROnPlateau,
        scheduler_kwargs=dict(
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
            ),
        ),
    v_00_52=dict(
        model_name='cnn_tierpsy',
        batch_size=128,
        learning_rate=1e-3,
        n_epochs=200,
        num_workers=4,
        scheduler=ReduceLROnPlateau,
        scheduler_kwargs=dict(
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
            ),
        ),
    v_01_00=dict(
        model_name='cnn_tierpsy_roi48',
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=120,
        num_workers=4,
        scheduler=None,
        ),
    v_01_01=dict(
        model_name='cnn_tierpsy_roi48',
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=120,
        num_workers=4,
        scheduler=ReduceLROnPlateau,
        scheduler_kwargs=dict(
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
            ),
        ),
    )


def train_fun(
        session_name,
        cuda_id=0,
        dataset_path=None,
        log_dir=None,
        ):

    # retrieve session parameters from the above list
    session_parameters = SESSIONS[session_name]

    # get paths
    dataset_path = sanitise_path(dataset_path, 'dataset')
    log_dir = sanitise_path(log_dir, 'log_dir')

    # set specified cuda device, or fall back on default gpu or cpu
    device = set_device(cuda_id)

    # get model and datasets
    model, criterion, (train_data, val_data) = get_model_datasets_criterion(
        session_parameters['model_name'],
        which_splits=['train', 'val'],
        data_path=dataset_path)
    # send to device, only then make optimiser
    model = model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=session_parameters['learning_rate'])
    # deal with learning rate scheduler
    if session_parameters['scheduler']:
        scheduler = session_parameters['scheduler'](
            optimiser,
            **session_parameters['scheduler_kwargs']
            )
    else:
        scheduler = None

    # use training function
    train_model(
        session_name,
        model,
        device,
        train_data,
        val_data,
        criterion,
        optimiser,
        log_dir,
        scheduler=scheduler,
        n_epochs=session_parameters['n_epochs'],
        batch_size=session_parameters['batch_size'],
        num_workers=session_parameters['num_workers'],
        )


if __name__ == '__main__':
    import fire
    fire.Fire(train_fun)
