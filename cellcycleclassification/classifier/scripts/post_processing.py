#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:20:15 2020

@author: lferiani
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay)

from torchvision.utils import make_grid

from cellcycleclassification.classifier.utils import get_default_log_dir
from cellcycleclassification.classifier.scripts.eval import (
    evaluate_performance_one_trained_model, model_path_to_name)



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


def meanperclassrecall(conf_mat):
    """
    From the confusion matrix output by sklearn, measure the recall in each
    class, and average across classes

    Parameters
    ----------
    conf_mat : np.ndarray
        in row i, see how many instances of label i were classified with
        label j (varies across columns)

    Returns
    -------
    mean_perclass_recall: float.

    """

    # sum across columns to just count how many images of each class
    # we tried to classify
    class_support = conf_mat.sum(axis=1)
    # nnumber of correctly classified is on the diagonal
    truly_belonging_to_class = conf_mat.diagonal()
    # divide "per class true positives" by total number
    perclass_recall = truly_belonging_to_class / class_support
    # average across classes
    mean_perclass_recall = perclass_recall.mean()
    return mean_perclass_recall


def xval_evaluation_wrapper(model_fname, dataset_fname):
    """
    Call evaluate_performance_one_trained_model(),
    sorts the output into a dataframe
    """
    (
     preds, labels, imgs, isfirstinstage, track_id, frame_number, probas,
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

    return work_df, train_pars, probas, imgs


def post_process(raw_df, probs, is_plot=False):
    """
    Take a df with the result of predicting the xval set, and do postprocessing
    """
    processed_df = []

    for track_name, track_df in raw_df.groupby(by='track_id'):

        # index by frame
        track_df = track_df.reset_index().set_index('frame')
        # raw forbidden moves (for stats later)
        track_df['raw_forbidden'] = detect_dubious(track_df['ypred'])
        # despike
        track_df['ypred_post'] = despike(track_df['ypred'])
        # find things still wrong
        track_df['post_spike'] = detect_spikes(track_df['ypred_post'])
        track_df['post_forbidden'] = detect_dubious(track_df['ypred_post'])

        # put the old index on, for output
        processed_df.append(track_df.reset_index().set_index('index'))

    # reassembled processed df
    processed_df = pd.concat(processed_df, axis=0, ignore_index=False)

    return processed_df


def plot_post_processing(post_df, probs, imgs):

    # dictionary for class names
    if probs.shape[1] == 4:
        label_dict = {0: 'G0/1', 1: 'S', 2: 'G2', 3: 'M'}

    for track_name, track_df in post_df.groupby(by='track_id'):

        # create a new index to make plotting easier (rename old for clarity)
        track_df = track_df.reset_index().rename(
            columns={'index': 'index_in_full_df'})
        # select the right probabilities,and the right images
        track_probs = probs[track_df['index_in_full_df'].values, :].T


        # plots unless we fixed everything
        if track_df['ypred_post'].equals(track_df['ytrue']):
            continue

        mislabelled_df = track_df.query('ypred_post != ytrue')
        mislabelled_imgs = imgs[mislabelled_df['index_in_full_df'].values]


        # make images into one-row grid
        track_imgs_grid = make_grid(
            torch.Tensor(mislabelled_imgs[:, None, ...]),
            nrow=mislabelled_imgs.shape[0],
            padding=2,
            normalize=True,
            pad_value=1).numpy().transpose(1, 2, 0)

        fig, axs = plt.subplots(2, 1, figsize=(18, 6))
        ax = axs[0]
        # plot matrix of probabilities
        ax.imshow(track_probs, aspect='auto', cmap='Blues_r')
        ax.set_xticks(track_df.index)
        ax.invert_yaxis()
        # and add the line plots on top
        track_df.plot(
            y='ytrue',
            color='tab:green',
            marker='.',
            markersize=12,
            linewidth=4,
            ax=ax)
        track_df.plot(
            y='ypred',
            color='tab:orange',
            marker='.',
            markersize=12,
            linewidth=3,
            ax=ax)
        track_df.plot(
            y='ypred_post',
            color='tab:purple',
            marker='.',
            markersize=12,
            linewidth=2,
            ax=ax)
        track_df.query('post_spike == True').plot(
            y='ypred_post',
            color='r',
            marker='o',
            markersize=8,
            linestyle='none',
            markerfacecolor='none',
            label='spike, post',
            ax=ax)
        track_df.query('post_forbidden == True').plot(
            y='ypred_post',
            marker='x',
            markersize=8,
            linestyle='none',
            color='r',
            label='forbidden, post',
            ax=ax)
        ax.set_title(track_name)
        ax.set_xticklabels(track_df['frame'].astype(int), rotation=45)
        ax.set_xlabel('frame')
        ax.set_yticks(range(probs.shape[1]))
        ax.set_yticklabels(label_dict[k] for k in ax.get_yticks())
        ax.set_ylabel('stage')

        ax = axs[1]
        ax.imshow(track_imgs_grid)
        ax.set_yticks([])
        roi_size = mislabelled_imgs.shape[1] + 2  # account for padding
        n_rois = mislabelled_imgs.shape[0]
        _xticks = np.arange(
            roi_size/2, (0.5 + n_rois) * roi_size, roi_size)
        ax.set_xticks(_xticks)
        ax.set_xticklabels(mislabelled_df['frame'].values.astype(int))

        fig.tight_layout()
        axs[0].legend(
            ncol=5,
            bbox_to_anchor=(1, 1),
            bbox_transform=fig.transFigure)

    return


def get_classification_stats(
        proc_df, truecol='ytrue', predcol='ypred', filter_col=None,
        is_plot=False, ax=None):
    # get data
    ground_truth = proc_df[truecol]
    predictions = proc_df[predcol]
    if filter_col:
        idx_filter = proc_df[filter_col]
        ground_truth = ground_truth[idx_filter]
        predictions = predictions[idx_filter]
    # classification reports
    crep = classification_report(
        ground_truth, predictions, output_dict=True)
    # confusion matrices
    cm = confusion_matrix(
        ground_truth, predictions)
    # mean per-class accuracy

    # plots
    if is_plot:
        if ax is None:
            fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=cm).plot(cmap='Blues', ax=ax)

    return crep


def are_predictions_perfect(df, truecol, predcol):
    return df[truecol].equals(df[predcol])


def percentage_perfect_predictions_tracks(df, truecol, predcol):
    """
    On each track, check if all the predictions match the ground truth.
    Then return eprcentage of tracks in which it happens
    """
    pct_perfect_tracks = df.groupby('track_id')[[truecol, predcol]].apply(
        are_predictions_perfect, truecol, predcol)
    # pct_perfect_tracks is an array of bools,
    # so percentage it's just its mean * 100
    pct_perfect_tracks = pct_perfect_tracks.mean() * 100
    return pct_perfect_tracks


def assess_postproc(model_fname, dataset_fname, is_plot=False):
    """
    Evaluate one epoch without shuffle or sampling,
    return dictionary with info to put in a file
    """
    # get further info, evaluate, post process, get other info
    model_name, training_session_name = model_path_to_name(model_fname)
    raw_df, train_pars, probas, imgs = xval_evaluation_wrapper(
        model_fname, dataset_fname)
    processed_df = post_process(raw_df, probas)
    if is_plot:
        plot_post_processing(processed_df, probas, imgs)
    # import pdb
    # pdb.set_trace()
    # measure goodness
    crep_raw = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred', is_plot=is_plot)
    pct_perfect_tracks_raw = percentage_perfect_predictions_tracks(
        processed_df, truecol='ytrue', predcol='ypred')
    pct_noforbidden_diffs_tracks_raw = (
        ~processed_df.groupby('track_id')['raw_forbidden'].any()
        ).mean() * 100
    crep_post = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred_post', is_plot=is_plot)
    pct_perfect_tracks_post = percentage_perfect_predictions_tracks(
        processed_df, truecol='ytrue', predcol='ypred_post')
    pct_noforbidden_diffs_tracks_post = (
        ~processed_df.groupby('track_id')['post_forbidden'].any()
        ).mean() * 100
    # same for first in stage
    crep_raw_fis = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred',
        filter_col='fis')
    crep_post_fis = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred_post',
        filter_col='fis')

    out = {
        'trained_model_name': model_name,
        'training_session': training_session_name,
        'model_type': train_pars['model_name'],
        'mpc_recall_raw_%':
            crep_raw['macro avg']['recall'] * 100,
        'mpc_precision_raw_%':
            crep_raw['macro avg']['precision'] * 100,
        'mpc_f1_raw_%':
            crep_raw['macro avg']['f1-score'] * 100,
        'accuracy_raw_%':
            crep_raw['accuracy'] * 100,
        'mpc_recall_postprocessed_%':
            crep_post['macro avg']['recall'] * 100,
        'mpc_precision_postprocessed_%':
            crep_post['macro avg']['precision'] * 100,
        'mpc_f1_postprocessed_%':
            crep_post['macro avg']['f1-score'] * 100,
        'accuracy_postprocessed_%':
            crep_post['accuracy'] * 100,
        'mpc_recall_raw_firstinstage_%':
            crep_raw_fis['macro avg']['recall'] * 100,
        'mpc_precision_raw_firstinstage_%':
            crep_raw_fis['macro avg']['precision'] * 100,
        'mpc_f1_raw_firstinstage_%':
            crep_raw_fis['macro avg']['f1-score'] * 100,
        'accuracy_raw_firstinstage_%':
            crep_raw_fis['accuracy'] * 100,
        'mpc_recall_postprocessed_firstinstage_%':
            crep_post_fis['macro avg']['recall'] * 100,
        'mpc_precision_postprocessed_firstinstage_%':
            crep_post_fis['macro avg']['precision'] * 100,
        'mpc_f1_postprocessed_firstinstage_%':
            crep_post_fis['macro avg']['f1-score'] * 100,
        'accuracy_postprocessed_firstinstage_%':
            crep_post_fis['accuracy'] * 100,
        '%_tracks_w/o_forbidden_steps_raw': pct_noforbidden_diffs_tracks_raw,
        '%_tracks_w/o_forbidden_steps_post': pct_noforbidden_diffs_tracks_post,
        '%_perfectly_predicted_tracks_raw': pct_perfect_tracks_raw,
        '%_perfectly_predicted_tracks_post': pct_perfect_tracks_post,
        'roi_size': train_pars['roi_size'],
        'is_use_sampler': train_pars['is_use_sampler'],
        }

    return out, processed_df


# %%
if __name__ == '__main__':

    # where are things
    data_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    # dataset_fname = data_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    dataset_fname = (
        data_dir
        / 'new_annotated_datasets'
        # / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'
        / 'R5C5F_PCNA_dl_dataset_20201216.hdf5'
        )
    model_dir = get_default_log_dir()

    # get list of files
    model_fnames = list(model_dir.rglob('*.pth'))
    model_fnames = [mf for mf in model_fnames if 'best' not in str(mf)]

    is_debug = True

    plt.close('all')

    if is_debug:
        vn = 'v_12_50'
        vn = 'v_19_53'
        vn = 'v_15_53'
        model_fnames = [mf for mf in model_fnames if vn in str(mf)]
        out, proc_df = assess_postproc(model_fnames[0], dataset_fname, is_plot=True)
    else:
        # no plot, just get numbers out
        out = []
        for model_fname in tqdm(model_fnames):
            out.append(assess_postproc(model_fname, dataset_fname)[0])
        # and write to csv
        pd.DataFrame(out).sort_values(by='training_session').to_csv(
            model_dir / 'reports' / 'postprocessing_performance.csv',
            float_format='%.3f')








