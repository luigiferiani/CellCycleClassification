#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:00:40 2021

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
from cellcycleclassification.classifier.scripts.post_processing import (
    eval_and_postproc, assess_postprocessed)
from cellcycleclassification import DL_DATASET_PATH

# %%

if __name__ == '__main__':

    # where are things
    # dataset_fname = DL_DATASET_PATH / 'R5C5F1_PCNA_sel_annotations.hdf5'
    dataset_fname = (
        DL_DATASET_PATH
        # / 'R5C5F_PCNA_dl_dataset_20201027.hdf5'
        # / 'R5C5F_PCNA_dl_dataset_20201216.hdf5'
        / 'Bergsneider_dl_dataset_20210802.hdf5'
        )

    model_dir = get_default_log_dir()  # points to local dir if no vpn
    # model_dir = model_dir.parent / (model_dir.name + '20201027')

    plt.close('all')

    # get list of files
    model_fnames = list(model_dir.rglob('*.pth'))
    # this is the best binary model and we'll keep using this
    bin_model_vn = 'v_06_60'
    bin_model_fname = [mf for mf in model_fnames
                       if bin_model_vn in str(mf) and 'best' not in str(mf)][0]
    # no plot, just get numbers out
    out = []
    for model_fname in tqdm(model_fnames):
        try:
            processed_df, info_dict = eval_and_postproc(
                model_fname, bin_model_fname, dataset_fname, is_plot=False)[:2]
            report = assess_postprocessed(
                processed_df, info_dict, is_plot=True)
            out.append(report)
        except AssertionError as EE:
            print(model_fname, EE)
    # and write to csv
    pd.DataFrame(out).sort_values(by='training_session').to_csv(
        model_dir / 'reports' / 'compare_postprocessing_performance.csv',
        float_format='%.3f', mode='a')

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
