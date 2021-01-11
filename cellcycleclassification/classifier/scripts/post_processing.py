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
        skip_plot = (skip_plot or
                     (track_df['ypred_or_dk'].equals(track_df['ypred_dk']) and
                      track_df['ypred_combi_dk'].equals(track_df['ypred_dk']))
                     )

        if skip_plot:
            if 'is_S' not in track_df.columns:
                continue
            else:
                if track_df['is_S'].equals(track_df['ytrue'] == 1):
                    continue

        mislabelled_df = track_df.query(
            ('(ypred_dk != ytrue) | '
             '(ypred_or_dk != ytrue) | '
             '(ypred_combi_dk != ytrue)'))
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
    model_name, training_session_name = model_path_to_name(model_fname)
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

    return processed_df, info_dict


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
    # model_dir = model_dir.parent / (model_dir.name + '20201027')

    # get list of files
    model_fnames = list(model_dir.rglob('*.pth'))
    # model_fnames = [mf for mf in model_fnames if 'best' not in str(mf)]

    # %%
    is_debug = True
    plt.close('all')

    # this is the best binary model and we'll keep using this
    bin_model_vn = 'v_06_60'
    bin_model_fname = [mf for mf in model_fnames
                       if bin_model_vn in str(mf) and 'best' in str(mf)][0]

    if is_debug:
        # select one model here manually
        vn = 'v_17_52_20201217_222014'
        # vn = 'v_12_63_20201218_213041'
        # vn = 'v_15_53_20201202_183728'  # best model for 20201027 dataset
        model_fname = [mf for mf in model_fnames if vn in mf.stem][0]
        # process multi, integrate with binary
        # very inefficient bc always using same binary...
        processed_df, info_dict = eval_and_postproc(
            model_fname, bin_model_fname, dataset_fname, is_plot=True)
        report = assess_postprocessed(processed_df, info_dict, is_plot=True)
        pprint(report)

    else:
        # no plot, just get numbers out
        out = []
        for model_fname in tqdm(model_fnames):
            try:
                processed_df, info_dict = eval_and_postproc(
                    model_fname, bin_model_fname, dataset_fname, is_plot=False)
                report = assess_postprocessed(
                    processed_df, info_dict, is_plot=True)
                out.append(report)
            except AssertionError as EE:
                print(model_fname, EE)
        # and write to csv
        pd.DataFrame(out).sort_values(by='training_session').to_csv(
            model_dir / 'reports' / 'compare_postprocessing_performance.csv',
            float_format='%.3f')

    # %%
    # read performance dataframe and rank
    performance_df = pd.read_csv(
        model_dir / 'reports' / 'compare_postprocessing_performance.csv')
    performance_df = performance_df.drop(columns='Unnamed: 0')
    performance_df = performance_df.filter(regex='^(?!bin)', axis=1)
    performance_df = performance_df.filter(regex='(?<!raw_%)$', axis=1)
    performance_df = performance_df.filter(regex='(?<!dk_%)$', axis=1)

    # select best 20 models
    meta_cols = [
        'trained_model_name', 'training_session', 'model_type',
        'roi_size', 'is_use_sampler']
    cols = [
        'accuracy_', 'mpc_precision_', 'mpc_recall_', 'mpc_f1_',
        'perfect_tracks_', 'tracks_w/o_forbidden_steps_']
    sufs = ['or_%', 'combi_%']
    top_df = []
    for col in [c+s for c in cols for s in sufs]:
        top_df.append(
            performance_df.query('is_use_sampler == True').sort_values(
                by=col, ascending=False).head(n=10)
        )
    top_df = pd.concat(top_df, axis=0, ignore_index=True).drop_duplicates()
    # %%
    print('or - combi, percentage points')
    for col in cols:
        print(col)
        or_col = col + 'or_%'
        combi_col = col + 'combi_%'
        frac_or_beats_combi = np.mean(top_df[or_col] - top_df[combi_col])
        print(frac_or_beats_combi)

    # %%
    # select best network for the two postprocessing methods
    best = {}
    for s in sufs:
        print(f'best for {s}')
        tmp_df = performance_df.query('is_use_sampler == True')
        data_cols = [c for c in tmp_df.columns if s in c]
        print(data_cols)
        # create a df with only ranks
        tmp_df_ranks = tmp_df[data_cols].rank(
            axis=0, method='dense', ascending=False)
        tmp_df['avg_rank'] = tmp_df_ranks.mean(axis=1)
        tmp_df['overall_rank'] = tmp_df['avg_rank'].rank()
        tmp_df = tmp_df.sort_values(by='overall_rank')
        best[s] = tmp_df.head()

    # %%

    def _cols_containing(df, ss):
        return [c for c in df.index if ss in c]

    best_or = best['or_%'].iloc[0]
    best_or = best_or[meta_cols + _cols_containing(best_or, 'or_%')]
    best_combi = best['combi_%'].iloc[0]
    best_combi = best_combi[
        meta_cols + _cols_containing(best_combi, 'combi_%')]

    print('best performing or')
    pprint(best_or)
    print('best performing combi')
    pprint(best_combi)
    print(
        best_or.filter(regex='(?<=%)$').values
        - best_combi.filter(regex='(?<=%)$').values
        )
    # basically the same, I think I'll implement the combi method as it
    # gives more tracks without issues

    # # we only care about _multi
    # performance_df = performance_df[
    #     performance_df['model_type'].str.contains('multi')]

    # data_cols = [c
    #              for c in performance_df.columns
    #              if (c.endswith('%') and ('post' in c) and ('first' not in c))
    #              ]

    # performance_df_ranks = performance_df[data_cols].rank(
    #     axis=0, method='min', ascending=False)

    # performance_df['avg_rank'] = performance_df_ranks.mean(axis=1)
    # performance_df['overall_rank'] = performance_df['avg_rank'].rank()

    # performance_df = performance_df.sort_values(by='overall_rank')

    # print(performance_df.head())




# %%
