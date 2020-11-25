#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:20:15 2020

@author: lferiani
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay)

from cellcycleclassification.classifier.utils import get_default_log_dir
from cellcycleclassification.classifier.scripts.eval import evaluate_performance_one_trained_model


def detect_spikes(signal_s):
    """
    A spike is a frame labelled as B while both the 2 neighbouring frames are A
    """
    is_spike = np.logical_and(
        signal_s.diff() == signal_s.diff(periods=-1),
        signal_s.diff() != 0)
    return is_spike


def detect_dubious(signal_s):
    """
    Detects if:
        diff is a weird number (should only ever be 0, 1 or -3)
    """
    # use diff the other way around, so value[n] - value[n-1]
    allowed_diffs = [np.nan, 0, -1, +3]
    is_forbidden_diff = ~signal_s.diff(periods=-1).isin(allowed_diffs)
    return is_forbidden_diff


def despike(signal_s):
    """
    find spikes (we find N)
    at most N times:
        go to the first spike, unspike it, find spikes again
    """

    def _rollmed(y):
        roll_med = y.rolling(3, center=True).median()
        roll_med[roll_med.isna()] = y[roll_med.isna()]
        return roll_med

    despiked = signal_s.copy()
    n_spikes_init = detect_spikes(despiked).sum()
    counter = 0
    n_spikes = n_spikes_init
    while n_spikes > 0 or counter == n_spikes_init:
        idx_tofix = detect_spikes(despiked)
        despiked[idx_tofix] = _rollmed(despiked)[idx_tofix]
        counter += 1
        n_spikes = detect_spikes(despiked).sum()

    return despiked.astype(signal_s.dtype)


def meanperclassaccuracy(confusion_matrix):
    """
    From the confusion matrix output by sklearn, measure the accuracy in each
    class, and average across classes

    Parameters
    ----------
    confusion_matrix : np.ndarray
        in row i, see how many instances of label i were classified with
        label j (varies across columns)

    Returns
    -------
    mean_perclass_accuracy: float.

    """

    # sum across columns to just count how many images of each class
    # we tried to classify
    class_support = confusion_matrix.sum(axis=1)
    # nnumber of correctly classified is on the diagonal
    correctly_classified = confusion_matrix.diagonal()
    # divide "per class true positives" by total number
    perclass_accuracy = correctly_classified / class_support
    # average across classes
    mean_perclass_accuracy = perclass_accuracy.mean()
    return mean_perclass_accuracy


def xval_evaluation_wrapper(model_fname, dataset_fname):
    """
    Call evaluate_performance_one_trained_model(),
    sorts the output into a dataframe
    """
    (
     preds, labels, imgs, isfirstinstage, track_id, frame_number,
     checkpoint, train_pars
     ) = evaluate_performance_one_trained_model(
         model_fname, dataset_fname, is_evaluate_xvalset=True)

    work_df = pd.DataFrame({
        'track_id': track_id,
        'frame': frame_number,
        'ytrue': labels,
        'ypred': preds,
        'fis': isfirstinstage})
    work_df = work_df.sort_values(by=['track_id', 'frame'])

    return work_df


def post_process(raw_df, is_plot=False):
    """
    Take a df with the result of predicting the xval set, and do postprocessing
    """
    processed_df = []
    for track_name, track_df in raw_df.groupby(by='track_id'):
        # index by frame
        track_df = track_df.reset_index().set_index('frame')
        # despike
        track_df['ypred_despiked'] = despike(track_df['ypred'])
        # find things still wrong
        track_df['pred_spike'] = detect_spikes(track_df['ypred_despiked'])
        track_df['forbidden_diff'] = detect_dubious(track_df['ypred_despiked'])

        # put the old index on, for output
        processed_df.append(track_df.reset_index().set_index('index'))

        # plots unless we fixed everything
        if track_df['ypred_despiked'].equals(track_df['ytrue']):
            continue
        if is_plot:
            fig, ax = plt.subplots()
            track_df.plot(y='ytrue', marker='.', ax=ax)
            track_df.plot(y='ypred', marker='.', ax=ax)
            track_df.plot(y='ypred_despiked', marker='.', ax=ax)
            track_df.query('pred_spike == True').plot(
                y='ypred_despiked',
                marker='o',
                linestyle='none',
                color='r',
                markerfacecolor='none',
                ax=ax)
            track_df.query('forbidden_diff == True').plot(
                y='ypred_despiked',
                marker='x',
                linestyle='none',
                color='r',
                ax=ax)
            ax.set_title(track_name)

    # reassembled processed df
    processed_df = pd.concat(processed_df, axis=0, ignore_index=False)

    return processed_df


def get_classification_stats(
        proc_df, truecol='ytrue', predcol='ypred', is_plot=False, ax=None):
    # classification reports
    crep = classification_report(
        proc_df[truecol], proc_df[predcol])
    # confusion matrices
    cm = confusion_matrix(
        proc_df[truecol], proc_df[predcol])
    # outputs
    print(f'{truecol} vs {predcol}')
    print(crep)
    print(f'Mean per-class accuracy, {predcol} = {meanperclassaccuracy(cm)}')
    # plots
    if is_plot:
        if ax is None:
            fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=cm).plot(cmap='Blues', ax=ax)

    return


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

    # model_fname = model_dir / 'v_14_60_20201121_231930/v_14_60_20201121_231930.pth'
    # model_fname = model_dir / 'v_14_51_20201121_232024/v_14_51_20201121_232024.pth'
    # model_fname = model_dir / 'v_15_60_20201121_231921/v_15_60_20201121_231921.pth'
    # model_fname = model_dir / 'v_13_50_20201121_231743/v_13_50_20201121_231743.pth'
    # model_fname = model_dir / 'v_13_60_20201121_231808/v_13_60_20201121_231808.pth'
    # model_fname = model_dir / 'v_11_50_20201120_194923/v_11_50_20201120_194923.pth'

    model_fnames = [
        'v_11_50_20201120_194923',
        'v_12_50_20201121_231643',
        'v_13_50_20201121_231743',
        # 'v_14_50_20201121_231859',
        # 'v_15_50_20201121_231910',
        ]

    model_fnames = [
        'v_11_60_20201120_195143',
        'v_12_60_20201121_231706',
        'v_13_60_20201121_231808',
        'v_14_60_20201121_231930',
        'v_15_60_20201121_231921',
        ]

    model_fnames = [model_dir / mf / (mf + '.pth') for mf in model_fnames]

    # %% plot all tracks if any ypred != ytrue
    plt.close('all')

    models_efforts_df = []
    for model_fname in model_fnames:
        raw_df = xval_evaluation_wrapper(model_fname, dataset_fname)
        processed_df = post_process(raw_df, is_plot=False)
        get_classification_stats(
            processed_df, truecol='ytrue', predcol='ypred')
        get_classification_stats(
            processed_df, truecol='ytrue', predcol='ypred_despiked')
        models_efforts_df.append(processed_df)
    # %%
    consensus_df = models_efforts_df[0].copy()
    consensus_df = consensus_df.rename(
        columns={'ypred_despiked': 'ypred_despiked_0'})
    for cc, df in enumerate(models_efforts_df):
        if cc == 0:
            continue
        consensus_df = pd.merge(
            consensus_df, df['ypred_despiked'],
            left_index=True, right_index=True)
        consensus_df = consensus_df.rename(
            columns={'ypred_despiked': f'ypred_despiked_{cc}'})
    # create consensus
    postproc_cols = [c for c in consensus_df.columns if c.startswith('ypred_')]
    consensus_df['consensus'] = consensus_df[postproc_cols].mode(axis=1)[0]

    get_classification_stats(
        consensus_df, truecol='ytrue', predcol='consensus', is_plot=True)

