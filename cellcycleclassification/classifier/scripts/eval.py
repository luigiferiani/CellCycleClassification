#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 18:56:06 2020

@author: lferiani
"""
import re
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from multiprocessing import Pool

# from functools import partial

# from pprint import pprint, pformat
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches
from matplotlib.widgets import TextBox
import matplotlib.gridspec as gridspec


from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from tensorboard.backend.event_processing import event_accumulator

import torch
import torchvision.utils

from cellcycleclassification.classifier.utils import (
    get_default_log_dir, get_training_parameters)
from cellcycleclassification.classifier.models.helper import (
    get_model_datasets_criterion)

from cellcycleclassification.classifier.models.helper import get_dataloader
from cellcycleclassification.classifier.trainer.engine import (
    evaluate_one_epoch)


# %%
def quickgrid(imgs, lbls, prdctns):

    pred_type = {
        (0, 0): 'True Neg',
        (1, 1): 'True Pos',
        (0, 1): 'False Pos',
        (1, 0): 'False Neg',
        }

    n_imgs = imgs.shape[0]
    grid_img = torchvision.utils.make_grid(
        imgs, padding=2, nrow=n_imgs,
        scale_each=True).numpy()[0, :, :]

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


def nicegrid(
        imgs,
        all_labels,  # all labels for dataset
        all_predictions,  # all predicitons for dataset
        isfirstinstage,  # bool True if first frame of a stage
        epochs_trained,
        training_parameters,
        log_path):  # how many epochs did the model train for

    # get a classification report
    is_multi = 'multi' in training_parameters['model_name']
    if is_multi:
        # class_id = [0, 1, 2, 3, 4]
        # class_labels = ['G0', 'G1', 'S', 'G2', 'M']
        class_id = [0, 1, 2, 3]
        class_labels = ['G0/1', 'S', 'G2', 'M']
    else:
        class_id = [0, 1]
        class_labels = ['not S', 'S']

    crep_all = classification_report(
        all_labels, all_predictions,
        labels=class_id, target_names=class_labels, output_dict=True)
    crep_first = classification_report(
        all_labels[isfirstinstage], all_predictions[isfirstinstage],
        labels=class_id, target_names=class_labels, output_dict=True)
    # import pdb
    # pdb.set_trace()

    # bin_pred_type = {  # second entry is prediction
    #     (0, 0): {'str': 'not S', 'c': 'g'},
    #     (1, 1): {'str': 'S', 'c': 'g'},
    #     (0, 1): {'str': 'S', 'c': 'r'},
    #     (1, 0): {'str': 'not S', 'c': 'r'},
    #     }

    # set up figure specs
    fig = plt.figure(figsize=(8.3, 11.7))
    gs = gridspec.GridSpec(
        nrows=3, ncols=2, left=0.04, right=0.92, bottom=0.01, top=0.98)

    # axis with text based info
    tax = fig.add_subplot(gs[0, 0])
    tax.set_visible(False)
    tbox = tax.get_position()
    # grect = [gbox.x0, gbox.y0, gbox.width, gbox.height]

    was_scheduler = training_parameters['scheduler'] is not None
    if was_scheduler:
        if training_parameters['scheduler_kwargs']['mode'] == 'max':
            scheduler_update_mode = 'max accuracy'
        elif training_parameters['scheduler_kwargs']['mode'] == 'min':
            scheduler_update_mode = 'min loss'
        else:
            upmode = training_parameters['scheduler_kwargs']['mode']
            raise ValueError(
                f'unknown sheduler update method {upmode}')
    else:
        scheduler_update_mode = None
    crep_txt = (
        f"All dataset:\n"
        f"Accuracy: {crep_all['accuracy']:.3f}\n"
        f"F1 score: {crep_all['S']['f1-score']:.3f}\n"
        f"Precision: {crep_all['S']['precision']:.3f}\n"
        f"Recall: {crep_all['S']['recall']:.3f}\n"
        f"\nFirst in stage:\n"
        f"Accuracy: {crep_first['accuracy']:.3f}\n"
        f"F1 score: {crep_first['S']['f1-score']:.3f}\n"
        f"Precision: {crep_first['S']['precision']:.3f}\n"
        f"Recall: {crep_first['S']['recall']:.3f}\n"
        f"\nModel name: {training_parameters['model_name']}\n"
        f"Epochs trained: {epochs_trained}\n"
        f"Batch size: {training_parameters['batch_size']}\n"
        f"Learning rate: {training_parameters['learning_rate']}\n"
        f"Used scheduler: {was_scheduler}\n"
        f"Scheduler update mode: {scheduler_update_mode}\n"
        f"Used sampler: {training_parameters['is_use_sampler']}\n"
        )

    fig.text(
        tbox.x0,
        tbox.y0 + tbox.height,
        crep_txt,
        verticalalignment='top',
        fontname='monospace',
        fontsize=12,
        wrap=True)

    # here goes the confusion matrix
    cmax = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(all_labels, all_predictions)
    cmdisp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels)
    cmdisp = cmdisp.plot(cmap='Blues', ax=cmax)

    # plot logs

    if log_path is not None:
        log_df = read_tb_log(log_path)
        lax = fig.add_subplot(gs[1, :])
        for metric in ['train_epoch_loss', 'val_epoch_loss']:
            log_df.plot(y=metric, ax=lax)
        lax.set_ylim(0, 1)

    # examples grid
    gax = fig.add_subplot(gs[2:, :])
    gax.set_visible(False)
    gbox = gax.get_position()
    grect = [gbox.x0, gbox.y0, gbox.width, gbox.height]

    # lets just use the "first in stage" as example
    imgs_first = imgs[isfirstinstage]
    labs_first = all_labels[isfirstinstage]
    preds_first = all_predictions[isfirstinstage]
    n_imgs = imgs_first.shape[0]
    if n_imgs > 18:
        n_imshows = 18
    else:
        n_imshows = n_imgs

    # find factors.
    factors = [i for i in range(1, n_imshows+1) if (n_imshows % i == 0)]
    n_imgs_per_row = factors[len(factors)//2]
    n_rows = n_imshows // n_imgs_per_row
    # make grid image
    grid = ImageGrid(
        fig,
        grect,
        nrows_ncols=(n_rows, n_imgs_per_row), axes_pad=0.1)
    for ax, im, ytrue, ypred in zip(
            grid,
            imgs_first[:n_imshows],
            labs_first[:n_imshows],
            preds_first[:n_imshows]):
        edge_color = 'g' if (ytrue == ypred) else 'r'
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
        ax.text(
            0.05, 0.05, class_labels[ypred],
            color='w',
            fontweight='bold',
            verticalalignment='top')
        ax.set_xticks([])
        ax.set_yticks([])
        rect = patches.Rectangle(
            (0, 0), im.shape[1], im.shape[0],
            edgecolor=edge_color,
            linewidth=2,
            facecolor='none',
            )
        ax.add_patch(rect)

    return fig


def find_log(model_fname):
    evs = list(model_fname.parent.rglob('events.out.tfevents*.*'))
    if len(evs) == 0:
        return None
    elif len(evs) > 1:
        warnings.warn('multiple logs found, returning the first')
    return evs[0]


def read_tb_log(log_path):
    ea = event_accumulator.EventAccumulator(
        str(log_path), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    df = []
    for metric in ea.Tags()['scalars']:
        _df = pd.DataFrame(ea.Scalars(metric))[['step', 'value']].rename(
            columns={'value': metric}).set_index('step')
        df.append(_df)

    df = pd.concat(df, axis=1)
    return df


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


def evaluate_performance_one_trained_model(
        model_fname, dataset_fname, is_evaluate_xvalset=False):
    """Run evaluation of an epoch of trained model."""
    # first get name and training session of model
    model_name, training_session_name = model_path_to_name(model_fname)
    # get the training parameters
    train_pars = get_training_parameters(training_session_name)
    # get model and validation dataset
    model, criterion, val_dataset = get_model_datasets_criterion(
        train_pars['model_name'],
        which_splits='val',
        data_path=dataset_fname,
        roi_size=train_pars['roi_size'])
    val_dataset.set_use_transforms(False)
    val_dataset.is_return_extra_info = True
    # create dataset/loader
    if is_evaluate_xvalset:
        # no oversampling, no shuffling
        val_loader = get_dataloader(
            val_dataset,
            False,
            train_pars['batch_size'],
            train_pars['num_workers'],
            is_use_shuffle=False,
            )
    else:
        # this is the same loader of the training
        val_loader = get_dataloader(
            val_dataset,
            train_pars['is_use_sampler'],
            train_pars['batch_size'],
            train_pars['num_workers'],
            )
    # import pdb
    # pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    checkpoint = torch.load(model_fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = checkpoint['epoch']
    # evaluate a full epoch
    _, _, prds, lbls, imgs, isfirstinstage, track_id, frame_number, prbs = (
        evaluate_one_epoch(
            model_name,
            model,
            criterion,
            val_loader,
            device,
            last_epoch,
            logger=None,
            is_return_extra_info=True,
            )
        )

    return (
        prds, lbls, imgs, isfirstinstage, track_id, frame_number, prbs,
        checkpoint, train_pars
        )


def evaluate_and_report_performance_one_trained_model(
        model_fname, dataset_fname, is_force_reeval=False):
    """Run evaluation of an epoch of trained model. Create report"""
    # first check for existence
    fig_savename = (
        model_fname.parent.parent
        / 'reports' / model_fname.with_suffix('.pdf').name
        )
    if fig_savename.exists() and not is_force_reeval:
        return

    # evaluate first
    all_eval_out = (
        evaluate_performance_one_trained_model(model_fname, dataset_fname)
        )
    predictions, labels, images, isfirstinstage, track_id, frame_number = (
        all_eval_out[:6])
    checkpoint, train_pars = all_eval_out[-2:]

    class_rep = classification_report(labels, predictions, output_dict=True)
    foo = classification_report(
        labels[isfirstinstage], predictions[isfirstinstage], output_dict=False)
    print('report for first of stage:')
    print(foo)

    log_path = find_log(model_fname)

    val_accuracy = class_rep['accuracy']
    # import pdb
    # pdb.set_trace()
    fig = nicegrid(
        images,
        labels,
        predictions,
        isfirstinstage,
        checkpoint['epoch'],  # last_epoch
        train_pars,
        log_path)
    fig.savefig(fig_savename)
    plt.close(fig)

    return (val_accuracy, model_fname, class_rep)



# %%

if __name__ == '__main__':

    # where are things
    data_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    # dataset_fname = data_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    dataset_fname = (
        data_dir
        / 'new_annotated_datasets'
        / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'
        )
    model_dir = get_default_log_dir()

    model_fnames = list(find_trained_models(model_dir))

    accs = []
    plt.ioff()
    for model_fname in tqdm(model_fnames):
        out = evaluate_and_report_performance_one_trained_model(
            model_fname, dataset_fname, is_force_reeval=True)
        accs.append(out)
    plt.ion()
    # out = evaluate_performance_one_trained_model(
    #     model_fnames[80], dataset_fname)

# %%
    # model_fname = model_dir / 'v_14_60_20201121_231930/v_14_60_20201121_231930.pth'

    # (
    #  preds, labels, imgs, isfirstinstage, track_id, frame_number,
    #  checkpoint, train_pars
    #  ) = evaluate_performance_one_trained_model(model_fname, dataset_fname, is_evaluate_xvalset=True)



