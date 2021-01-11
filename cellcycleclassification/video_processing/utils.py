#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Jan 11 16:33:03 2021

@author: lferiani
"""
import numpy as np
import pandas as pd


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


def despike_signal(signal_s):
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


def despike(raw_df, col='ypred', col_post='ypred_dk'):
    """
    Take a df with the result of predicting the xval set, and do postprocessing
    """
    processed_df = []
    for track_name, track_df in raw_df.groupby(by='track_id'):
        # index by frame
        track_df = track_df.reset_index().set_index('frame')
        # despike
        track_df[col_post] = despike_signal(track_df[col])
        # put the old index on, for output
        processed_df.append(track_df.reset_index().set_index('index'))

    # reassembled processed df
    processed_df = pd.concat(processed_df, axis=0, ignore_index=False)

    return processed_df


def find_forbidden_transitions(df, col='ypred', col_forbidden='raw_forbidden'):
    """
    Find forbidden transitions in the entire dataframe
    """

    out_df = []
    for track_name, track_df in df.groupby(by='track_id'):

        # index by frame
        track_df = track_df.reset_index().set_index('frame')
        # raw forbidden moves (for stats later)
        track_df[col_forbidden] = detect_dubious(track_df[col])
        # put the old index on, for output
        out_df.append(track_df.reset_index().set_index('index'))

    # reassembled processed df
    out_df = pd.concat(out_df, axis=0, ignore_index=False)

    return out_df


def get_model_children(model):
    import torch.nn as nn
    keep_list = [nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ReLU]
    # initialise
    # model_weights = []
    conv_layers = []
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for child in model_children:
        if type(child) in keep_list:
            counter += 1
            # model_weights.append(child.weight)
            conv_layers.append(child)
        elif type(child) == nn.Sequential:
            for gchild in child:
                if type(gchild) in keep_list:
                    counter += 1
                    # model_weights.append(gchild.weight)
                    conv_layers.append(gchild)
    print(f"Total convolutional layers: {counter}")
    # return model_weights, conv_layers
    return conv_layers


def through_unpacked_layers(img, conv_layers):
    with torch.no_grad():
        # pass the image through all the layers
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]).detach())
        # make a copy of the `results`
        outputs = results.copy()
    return outputs


def plot_featmaps(featmaps):
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    for layer in featmaps:
        # deal with useless minibatch
        featslist = [im[None, :, :] for im in layer[0]]
        # get how many plots per row
        n_plots = len(featslist)
        factors = [i for i in range(1, n_plots+1) if (n_plots % i == 0)]
        n_imgs_per_row = n_plots // factors[len(factors)//2]
        img_grid = make_grid(
            featslist,
            nrow=n_plots//n_imgs_per_row,
            pad_value=np.nan,
            padding=1,
            ).detach().numpy()
        img_grid = img_grid[0]
        # import pdb
        # pdb.set_trace()
        plt.figure()
        plt.imshow(img_grid, cmap='Blues')
        plt.tight_layout()


# if __name__ == "__main__":
#     from pathlib import Path

#     work_dir = Path('/Users/lferiani/work_repos/CellCycleClassification/data')
#     csv_fname = work_dir / 'R5C5F1_PCNA_sel.csv'
#     imgs_dir = work_dir / 'R5C5F1_PCNA/'
#     models_dir = Path(
#         '/Volumes/behavgenom$/Luigi/Data/AlexisBarr_cell_cycle_classification/'
#         'trained_models')
#     # model_path /= 'v_04_60_20200908_160037/v_04_60_best.pth'
#     bin_model_path = (
#         models_dir / 'v_06_60_20201217_113641/v_06_60_best.pth')

#     vidproc = VideoProcessor(
#         tracking_csv=csv_fname,
#         images_dir=imgs_dir,
#         model_fname=bin_model_path,
#         )
#     vidproc.process_video()
#     vidproc.export_csv()
#     vidproc.export_frames()

#     model_weights, conv_layers = get_model_children(vidproc.model)
#     conv_layers = get_model_children(vidproc.model)
#     roi_data = vidproc.get_rois_of_track_id(5)
#     # img = torch.from_numpy(roi_data[87][None, :, :, :])
#     img = torch.from_numpy(roi_data[60][None, :, :, :])
#     featmaps = through_unpacked_layers(img, conv_layers)

#     from matplotlib import pyplot as plt
#     plt.close('all')
#     plot_featmaps(featmaps)


#     im = vidproc.frame[0]

#     ifd = IndexedFigsDir(imgs_dir)
#     img = ifd[0]