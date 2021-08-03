#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:20:15 2020

@author: lferiani
"""
# %%

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
from cellcycleclassification.video_processing.utils import (
    despike, find_forbidden_transitions)


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


def plot_post_processing(post_df, probs, imgs):

    # dictionary for class names
    if probs.ndim == 1:
        # this was binary probability but make it 2d
        probs = np.concatenate(
            (1-probs[:, None], probs[:, None]), axis=1)
        label_dict = {0: 'not S', 1: 'S'}
    elif probs.shape[1] == 4:
        label_dict = {0: 'G0/1', 1: 'S', 2: 'G2', 3: 'M'}
    else:
        raise Exception('probs is of wrong size')

    for track_name, track_df in post_df.groupby(by='track_id'):

        # create a new index to make plotting easier (rename old for clarity)
        track_df = track_df.reset_index().rename(
            columns={'index': 'index_in_full_df'})
        # select the right probabilities,and the right images
        track_probs = probs[track_df['index_in_full_df'].values, :].T

        # plots unless we fixed everything
        skip_plot = all([track_df[col].equals(track_df['ytrue'])
                        for col in track_df.columns
                        if ('ypred_' in col) and ('_dk' in col)])
        if all(c in track_df for c in ['ypred_or_dk', 'ypred_combi_dk']):
            skip_plot = (
                skip_plot or
                (track_df['ypred_or_dk'].equals(track_df['ypred_dk']) and
                 track_df['ypred_combi_dk'].equals(track_df['ypred_dk']))
                )

        if skip_plot:
            if 'is_S' not in track_df.columns:
                continue
            else:
                if track_df['is_S'].equals(track_df['ytrue'] == 1):
                    continue

        mislabelled_query_str = '(ypred_dk != ytrue)'
        if 'ypred_or_dk' in track_df.columns:
            mislabelled_query_str += ' | (ypred_or_dk != ytrue)'
        if 'ypred_combi_dk' in track_df.columns:
            mislabelled_query_str += ' | (ypred_combi_dk != ytrue)'

        mislabelled_df = track_df.query(mislabelled_query_str)

        if 'is_S' in track_df.columns:
            mislabelled_df = pd.concat(
                (mislabelled_df,
                 track_df[track_df['is_S'] != (track_df['ytrue'] == 1)]),
                axis=0
                )
        mislabelled_imgs = imgs[mislabelled_df['index_in_full_df'].values]

        # make images into one-row grid
        try:
            track_imgs_grid = make_grid(
                torch.Tensor(mislabelled_imgs[:, None, ...]),
                nrow=mislabelled_imgs.shape[0],
                padding=2,
                normalize=True,
                pad_value=1).numpy().transpose(1, 2, 0)
        except:
            import pdb
            pdb.set_trace()

        # some commons line/marker specs
        forbidden_plot_specs = {
            'marker': 'x', 'markersize': 8, 'linestyle': 'none', 'color': 'r'}
        pp_line_specs = {'marker': '.', 'markersize': 12, }

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
            linewidth=6,
            ax=ax)
        track_df.plot(
            y='ypred',
            color='tab:orange',
            marker='.',
            markersize=12,
            linewidth=5,
            ax=ax)
        track_df.plot(
            y='ypred_dk',
            color='tab:purple',
            **pp_line_specs,
            linewidth=4,
            ax=ax)
        track_df.query('despiked_forbidden == True').plot(
            y='ypred_dk',
            **forbidden_plot_specs,
            label='forbidden, despiked',
            ax=ax)
        if 'ypred_or_dk' in track_df.columns:
            track_df.plot(
                y='ypred_or_dk',
                color='tab:brown',
                linewidth=3,
                **pp_line_specs,
                ax=ax)
            track_df.query('or_forbidden == True').plot(
                y='ypred_or_dk',
                **forbidden_plot_specs,
                label='forbidden, or',
                ax=ax)
        if 'ypred_combi_dk' in track_df.columns:
            track_df.plot(
                y='ypred_combi_dk',
                color='tab:cyan',
                linewidth=2,
                **pp_line_specs,
                ax=ax)
            track_df.query('combi_forbidden == True').plot(
                y='ypred_combi_dk',
                **forbidden_plot_specs,
                label='forbidden, combi',
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


def tracks_longer_than_one_frame(df):
    # probably a one-liner, but I want it to be legible
    tids = [tid for tid, tdf in df.groupby('track_id') if len(tdf) > 1]
    return tids


def percentage_perfect_predictions_tracks(
        df, truecol, predcol, include_oneframe_tracks=False):
    """
    On each track, check if all the predictions match the ground truth.
    Then return percentage of tracks in which it happens
    """
    if not include_oneframe_tracks:
        multiframe_tracks = tracks_longer_than_one_frame(df)
        _df = df[df['track_id'].isin(multiframe_tracks)]
    else:
        _df = df
    pct_perfect_tracks = _df.groupby('track_id')[[truecol, predcol]].apply(
        are_predictions_perfect, truecol, predcol)
    # pct_perfect_tracks is an array of bools,
    # so percentage it's just its mean * 100
    pct_perfect_tracks = pct_perfect_tracks.mean() * 100
    return pct_perfect_tracks


def percentage_no_forbidden_transitions_tracks(
        df, forbiddencol, include_oneframe_tracks=False):
    if not include_oneframe_tracks:
        multiframe_tracks = tracks_longer_than_one_frame(df)
        print(f'selecting {len(multiframe_tracks)} ')
        print(f'out of {len(df["track_id"].unique())}')
        _df = df[df['track_id'].isin(multiframe_tracks)]
    else:
        _df = df
    # each track, does it have a forbidden transition? 1=yes, 0=no. invert it
    # fraction is mean, percentage is *100
    return (~_df.groupby('track_id')[forbiddencol].any()).mean() * 100


def eval_and_postproc(
        multi_model_fname, binary_model_fname, dataset_fname, is_plot=False):
    """
    Evaluate one epoch without shuffle or sampling,
    return dictionary with info to put in a file
    """
    # get further info, evaluate, post process, get other info
    model_name, training_session_name = model_path_to_name(multi_model_fname)
    # evaluate multilabel model
    raw_df, train_pars, probas, imgs = xval_evaluation_wrapper(
        multi_model_fname, dataset_fname)
    # check that multi is indeed multi
    assert 'multi' in train_pars['model_name'], 'not a multilabel CNN'
    # same for binary
    bin_model_name, bin_training_session_name = model_path_to_name(
        binary_model_fname)
    bin_raw_df, bin_train_pars, bin_probas, bin_imgs = xval_evaluation_wrapper(
        binary_model_fname, dataset_fname)

    # dictionary for output info
    info_dict = {
        'trained_model_name': model_name,
        'training_session': training_session_name,
        'model_type': train_pars['model_name'],
        'roi_size': train_pars['roi_size'],
        'is_use_sampler': train_pars['is_use_sampler'],
        'bin_trained_model_name': bin_model_name,
        'bin_training_session': bin_training_session_name,
        'bin_model_type': bin_train_pars['model_name'],
        'bin_roi_size': bin_train_pars['roi_size'],
        'bin_is_use_sampler': bin_train_pars['is_use_sampler'],
    }

    # do simple despiking
    despiked_df = despike(raw_df, col='ypred', col_post='ypred_dk')
    bin_despiked_df = despike(bin_raw_df, col='ypred', col_post='ypred_dk')

    # combine despiked so if multilabel is S or bin label is S => label is S
    # and then despike further
    processed_df = despiked_df.copy(deep=True)
    processed_df['ypred_or'] = processed_df['ypred_dk']
    processed_df.loc[bin_despiked_df['ypred_dk'] == 1, 'ypred_or'] = 1
    processed_df = despike(
        processed_df, col='ypred_or', col_post='ypred_or_dk')

    # alternative method, combinining raw probabilities
    n_labels = probas.shape[1]
    bin_probas_expanded = np.array([(1-bin_probas)/(n_labels-1), bin_probas])
    bin_probas_expanded = bin_probas_expanded[[0, 1, 0, 0], :].T
    assert bin_probas_expanded.shape[1] == n_labels
    probas_combined = bin_probas_expanded/2 + probas/2
    # make predictions and add to the processed dataframe
    # (dataframe is sorted by track and time)
    predictions_combined = np.argmax(probas_combined, axis=1)
    processed_df['ypred_combi'] = predictions_combined[processed_df.index]
    processed_df = despike(
        processed_df, col='ypred_combi', col_post='ypred_combi_dk')

    # find all dubious transitions
    processed_df = find_forbidden_transitions(
        processed_df, col='ypred', col_forbidden='raw_forbidden')
    processed_df = find_forbidden_transitions(
        processed_df, col='ypred_dk', col_forbidden='despiked_forbidden')
    processed_df = find_forbidden_transitions(
        processed_df, col='ypred_or_dk', col_forbidden='or_forbidden')
    processed_df = find_forbidden_transitions(
        processed_df, col='ypred_combi_dk', col_forbidden='combi_forbidden')

    if is_plot:
        plot_post_processing(processed_df, probas, imgs)

    return processed_df, info_dict, raw_df, bin_raw_df, probas_combined, imgs
    # return processed_df, info_dict, raw_df, bin_raw_df


def assess_one_method(
        df, truecol='ytrue', predcol='ypred', forbiddencol='raw_forbidden',
        suffix='raw',
        is_plot=False):
    """
    Get classification report, plus % perfect tracks and tracks without
    forbidden transitions
    """
    crep = get_classification_stats(
        df, truecol=truecol, predcol=predcol, is_plot=is_plot)
    crep['%_perfect_tracks'] = percentage_perfect_predictions_tracks(
        df, truecol=truecol, predcol=predcol, include_oneframe_tracks=False)
    crep['%_nobadtrans_tracks'] = percentage_no_forbidden_transitions_tracks(
        df, forbiddencol=forbiddencol, include_oneframe_tracks=False)

    # make output better
    out = {
        f'mpc_recall_{suffix}_%': crep['macro avg']['recall'] * 100,
        f'mpc_precision_{suffix}_%': crep['macro avg']['precision'] * 100,
        f'mpc_f1_{suffix}_%': crep['macro avg']['f1-score'] * 100,
        f'accuracy_{suffix}_%': crep['accuracy'] * 100,
        f'perfect_tracks_{suffix}_%': crep['%_perfect_tracks'],
        f'tracks_w/o_forbidden_steps_{suffix}_%': crep['%_nobadtrans_tracks']
    }

    return out


def assess_postprocessed(processed_df, info_dict, is_plot=False):

    # measure goodness of raw multi classification
    crep_raw = assess_one_method(
        processed_df,
        truecol='ytrue',
        predcol='ypred',
        forbiddencol='raw_forbidden',
        suffix='raw',
        is_plot=is_plot)
    # despiked multi classification
    crep_dk = assess_one_method(
        processed_df,
        truecol='ytrue',
        predcol='ypred_dk',
        forbiddencol='despiked_forbidden',
        suffix='dk',
        is_plot=is_plot)
    # multi | binary classification
    crep_or = assess_one_method(
        processed_df,
        truecol='ytrue',
        predcol='ypred_or_dk',
        forbiddencol='or_forbidden',
        suffix='or',
        is_plot=is_plot)
    # multi | binary classification
    crep_combi = assess_one_method(
        processed_df,
        truecol='ytrue',
        predcol='ypred_combi_dk',
        forbiddencol='combi_forbidden',
        suffix='combi',
        is_plot=is_plot)

    out = {**info_dict, **crep_raw, **crep_dk, **crep_or, **crep_combi}

    return out


# %%
if __name__ == '__main__':
    from cellcycleclassification import (
            BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH)

    # where are things
    data_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    # dataset_fname = data_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    dataset_fname = (
        data_dir
        / 'new_annotated_datasets'
        # / 'R5C5F_PCNA_dl_dataset_20201216.hdf5'
        / 'Bergsneider_dl_dataset_20210802.hdf5'
        )

    plt.close('all')

    # postprocessing
    processed_df, info_dict, multi_df, bin_df, probas_combined, imgs = (
        eval_and_postproc(
            MULTICLASS_MODEL_PATH, BINARY_MODEL_PATH, dataset_fname,
            is_plot=False)
        )
    report = assess_postprocessed(processed_df, info_dict, is_plot=True)
    pprint(report)

    processed_df = processed_df.drop(
        columns=['ypred_or', 'ypred_or_dk', 'or_forbidden'])
    plot_post_processing(processed_df, probas_combined, imgs)


# %% for figures purposes, show how well each model behaves

    savedir = (
        Path.home() /
        'OneDrive - Imperial College London/Slides/20210809_group_meeting')
    # bin
    fig, ax = plt.subplots()
    crep_bin = get_classification_stats(
        bin_df, truecol='ytrue', predcol='ypred', filter_col=None,
        is_plot=True, ax=ax)
    ax.set_xticklabels(['not S', 'S'])
    ax.set_yticklabels(['not S', 'S'])
    fig.tight_layout()
    fig.savefig(savedir / 'bin_confusion_matrix.pdf')

    # multi
    fig, ax = plt.subplots()
    crep_multi = get_classification_stats(
        multi_df, truecol='ytrue', predcol='ypred', filter_col=None,
        is_plot=True, ax=ax)
    ax.set_xticklabels(['G0/1', 'S', 'G2', 'M'])
    ax.set_yticklabels(['G0/1', 'S', 'G2', 'M'])
    fig.tight_layout()
    fig.savefig(savedir / 'multi_confusion_matrix.pdf')

    # just probabilities combination
    fig, ax = plt.subplots()
    crep_combi = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred_combi', filter_col=None,
        is_plot=True, ax=ax)
    ax.set_xticklabels(['G0/1', 'S', 'G2', 'M'])
    ax.set_yticklabels(['G0/1', 'S', 'G2', 'M'])
    fig.tight_layout()
    fig.savefig(savedir / 'combi_raw_confusion_matrix.pdf')

    # despiked probabilities combination
    fig, ax = plt.subplots()
    crep_combi_dk = get_classification_stats(
        processed_df, truecol='ytrue', predcol='ypred_combi_dk', filter_col=None,
        is_plot=True, ax=ax)
    ax.set_xticklabels(['G0/1', 'S', 'G2', 'M'])
    ax.set_yticklabels(['G0/1', 'S', 'G2', 'M'])
    fig.tight_layout()
    fig.savefig(savedir / 'combi_dk_confusion_matrix.pdf')

# %%
