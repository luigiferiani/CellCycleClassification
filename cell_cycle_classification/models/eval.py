#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 18:56:06 2020

@author: lferiani
"""

import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torchvision.utils
from torch.utils.data import DataLoader

from datasets import CellsDataset
from models import CNN_tierpsy

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
    plt.show()

# %%

if __name__ == '__main__':

    # where are things
    work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    dataset_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    model_fname = work_dir / 'CNN_tierpsy_state_20200828_171536.pth'

    # parameters
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device in use: {device}')

    # create dataset/loader
    val_data = CellsDataset(dataset_fname, which_set='val')
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=batch_size, num_workers=4)

    # instantiate the model and optimizer
    model = CNN_tierpsy().to(device)

    # load model
    checkpoint = torch.load(model_fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = checkpoint['epoch']
    training_loss = checkpoint['loss']
    # set in eval only
    model.eval()

    # evaluate:
    # lazy accumulation of results
    labels = []
    predictions = []
    is_hydras = []
    with torch.no_grad():
        for batch in val_loader:
            batch_imgs, batch_labels = batch
            # only images to device, labels don't matter
            batch_imgs = batch_imgs.to(device)
            # model does *not* return softmax
            logits = model(batch_imgs)
            # could use softmax but doesn't matter, pred is index of max logit
            batch_predictions = torch.argmax(logits, axis=1)
            predictions.append(batch_predictions)
            labels.append(batch_labels)

    imshow(batch_imgs, batch_labels, batch_predictions)

    # concatenate accumulators into np arrays for ease of use
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0).squeeze()

    report = classification_report(labels, predictions)
    conf_mat = confusion_matrix(labels, predictions, normalize='all')
    tn, fp, fn, tp = conf_mat.flatten()

