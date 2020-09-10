#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 18:56:06 2020

@author: lferiani
"""
import re
import numpy as np

from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torchvision.utils

from cellcycleclassification.classifier.utils import get_default_log_dir
from cellcycleclassification.classifier.scripts.train import SESSIONS
from cellcycleclassification.classifier.models.helper import (
    get_model_datasets_criterion)

from cellcycleclassification.classifier.models.helper import get_dataloader
from cellcycleclassification.classifier.trainer.engine import evaluate_one_epoch

# %%
def imshow(imgs, lbls, prdctns):

    pred_type = {
        (0, 0): 'True Neg',
        (1, 1): 'True Pos',
        (0, 1): 'False Pos',
        (1, 0): 'False Neg',
        }
    n_imgs = imgs.shape[0]
    grid_img = torchvision.utils.make_grid(
        imgs, padding=2, nrow=n_imgs,
        scale_each=False).numpy()[0, :, :]

    fig, ax = plt.subplots()
    ax.imshow(grid_img, cmap='gray')
    xlims = ax.get_xlim()
    spacing = (xlims[1]-xlims[0]) / n_imgs
    offset = spacing/2
    ticks = np.arange(xlims[0]+offset, xlims[1], spacing)
    ticklabels = [pred_type[(lab, pred)]
                  for lab, pred in zip(lbls.numpy(), prdctns.numpy())]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def find_trained_models(dirname):
    """Scan dirname, find trained models"""
    model_paths_gen = dirname.rglob('*.pth')
    return model_paths_gen


def model_path_to_name(model_path):
    """Take a model path, return model name and training session"""
    regex_endings = r'\_best$|\_\d{8}\_\d{6}$'
    model_name = model_path.stem
    training_session_name = re.sub(regex_endings, '', model_name)
    return model_name, training_session_name


def get_training_parameters(session_name):
    try:
        pars_dict = SESSIONS[session_name]
    except KeyError:
        print(f'cannot find parameters for {session_name}')
        return
    # patch parameters that were added later
    if 'is_use_sampler' not in pars_dict.keys():
        pars_dict['is_use_sampler'] = False
    return pars_dict


def evaluate_performance_one_trained_model(model_fname, dataset_fname):
    """Run evaluation of an epoch of trained model. Create report"""
    # first get name and training session of model
    model_name, training_session_name = model_path_to_name(model_fname)
    # get the training parameters
    train_pars = get_training_parameters(training_session_name)
    # get model and validation dataset
    model, criterion, val_dataset = get_model_datasets_criterion(
        train_pars['model_name'],
        which_splits='val',
        data_path=dataset_fname)
    val_dataset.set_use_transforms(False)
    # create dataset/loader
    val_loader = get_dataloader(
        val_dataset,
        train_pars['is_use_sampler'],
        train_pars['batch_size'],
        train_pars['num_workers']
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    checkpoint = torch.load(model_fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = checkpoint['epoch']
    # evaluate a full epoch
    with torch.no_grad():
        model.eval()
        val_loss = defaultdict(float)
        labels = defaultdict(torch.tensor)
        predictions = defaultdict(torch.tensor)
        # for mbc, data in enumerate(pbar):
        for mbc, data in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]
            batch_imgs, batch_labels = data[0].to(device), data[1].to(device)
            # forwards only
            out = model(batch_imgs)
            _loss = criterion(out, batch_labels)
            if out.ndim > 1:
                batch_predictions = torch.argmax(out, axis=1)
            else:
                batch_predictions = (torch.sigmoid(out) > 0.5).long()
            # store labels and predictions
            labels[mbc] = batch_labels.cpu()
            predictions[mbc] = batch_predictions.cpu()
            # store mini batch loss in accumulator
            val_loss[mbc] = _loss.item()
    # average or concatenate batch values
    val_loss = np.mean([val_loss[mbc] for mbc in val_loss.keys()])
    predictions = np.concatenate(
        [predictions[mbc].squeeze() for mbc in predictions.keys()])
    labels = np.concatenate(
        [labels[mbc].squeeze() for mbc in labels.keys()])

    class_rep = classification_report(labels, predictions, output_dict=True)
    val_accuracy = class_rep['accuracy']
    imshow(batch_imgs, batch_labels, batch_predictions)
    return class_rep


# %%

if __name__ == '__main__':

    # where are things
    data_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    dataset_fname = data_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    model_dir = get_default_log_dir()
    model_fname = model_dir / 'v_03_50_20200908_122535/v_03_50_best.pth'

    evaluate_performance_one_trained_model(model_fname, dataset_fname)












    # measures





