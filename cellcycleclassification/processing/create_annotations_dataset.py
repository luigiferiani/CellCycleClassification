#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:17:11 2020

@author: lferiani

This code creates an hdf5 ready to be manually annotated using the annotator ui.
Starts from a nuclitrack csv file and a folder of tiffs
Saves the hdf5 dataset in the same folder as the tracking csv
"""

import re
import cv2
import fire
import tables
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)


def read_nuclitrack_data(csv_fname):
    # few columns that should be integers
    INT_COLS = ['track_id', 'frame', 'parent_track_id', 'event_flag',
                'tree_label']
    # read csv, fix colnames, fix coltypes
    raw_df = pd.read_csv(csv_fname)
    raw_df.columns = (raw_df.columns
                            .str.strip()
                            .str.lower()
                            .str.replace(' ', '_'))
    for col in INT_COLS:
        raw_df[col] = raw_df[col].astype(int)

    # tracks can be lost for a few frames. Try interpolating the missing frames

    # this only inserts rows for the missing frames,
    # but does not do any interpolation
    int_df = []
    # group by track, keeping their order
    for gn, g in raw_df.groupby(by='track_id', sort=False):
        # create a continuous range of frames from start to end of track
        min_frame = g['frame'].min()
        max_frame = g['frame'].max()
        all_frames = range(min_frame, max_frame+1)
        # create new df without missing frames
        int_g = (g.reset_index(drop=True)
                  .set_index('frame')
                  .reindex(all_frames)
                  .reset_index(drop=False)
                 )
        # create another column with flag is_interpolated
        int_g['is_interpolated'] = int_g['track_id'].isna()
        # now fill the track_id
        int_g['track_id'] = int_g['track_id'].fillna(method='ffill')
        # linear interpolation for x and y
        int_g['x_center'] = int_g['x_center'].interpolate(
            method='linear', limit_area='inside')
        int_g['y_center'] = int_g['y_center'].interpolate(
            method='linear', limit_area='inside')
        # append to growing list of track dataframes
        int_df.append(int_g)

    # now make it a single dataframe
    int_df = pd.concat(int_df, axis=0, ignore_index=True, sort=False)

    # and add a column for labels
    int_df['label_id'] = 0

    return int_df


def parse_frame_number(input_path):
    pattern = r"(?<=sk)\d+(?=fk)"
    return int(re.findall(pattern, input_path.name)[0])-1


def get_ordered_imgs_list(imgs_dir):
    # scan the imgs_dir for images
    imgs_list = list(imgs_dir.rglob('*.tiff'))
    # get frame number by parsing filename (rglob does not get stuff in order)
    imgs_fnames = [(parse_frame_number(path), path) for path in imgs_list]
    # sort by frame number
    imgs_fnames = sorted(imgs_fnames)

    return imgs_fnames


def plot_dataset(dataset_fname):

    # retrieve data
    anns_df = pd.read_hdf(
        dataset_fname, key='/annotations_df')
    with tables.File(dataset_fname, 'r') as fid:
        imgs = fid.get_node('/full_data').read()

    # plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 12))
    for fn, img in enumerate(imgs):
        ax.imshow(img, cmap='gray')
        # idx_toplot = (tracks_df['frame'] > fn-5) & (tracks_df['frame'] <= fn)
        idx_toplot = anns_df['frame'] == fn
        df_toplot = anns_df[idx_toplot]
        ax.scatter(df_toplot['x_center'],
                   df_toplot['y_center'],
                   s=5 * ((~df_toplot['is_interpolated']).astype(int)*5+1),
                   c=df_toplot['track_id'])
        plt.draw()
        plt.pause(1)
        plt.cla()


def create_dataset(tracking_csv: Path, images_dir: Path):
    """
    create_dataset

    Convert the tracking data from nuclitrack and a folder of tiffs
    into a dataset that works with the annotator gui.
    The dataset is saved in the same folder as `tracking_csv`.

    Parameters
    ----------
    tracking_csv : Path
        Path to the nuclitrack csv.
    images_dir : Path
        Folder containing the tiffs.

    Returns
    -------
    None.

    """
    # sanitise input
    if isinstance(tracking_csv, str):
        tracking_csv = Path(tracking_csv)
    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    # create output name
    output_hdf5 = (tracking_csv.parent
                   / (tracking_csv.stem + '_annotations.hdf5'))
    # get data
    tracks_df = read_nuclitrack_data(tracking_csv)
    figslist = get_ordered_imgs_list(images_dir)
    # load images. This can be bad if too many of them
    imgs = [cv2.imread(str(img_fname), -1)[None, :, :]
            for _, img_fname in figslist]
    imgs = np.concatenate(imgs, axis=0)
    # write data in output
    # annotatios
    tracks_df.to_hdf(output_hdf5,
                     key='/annotations_df',
                     mode='w')
    # and images
    with tables.File(output_hdf5, 'a') as fid:
        fid.create_earray('/',
                          'full_data',
                          obj=imgs,
                          filters=FILTERS)

    return


def main():
    fire.Fire(create_dataset)


if __name__ == '__main__':
    main()

# if __name__ == '__main__':

#     # where are things:
#     work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
#     trackingdata_fname = work_dir / 'R5C5F1_PCNA_sel.csv'
#     imgs_dir = work_dir / 'R5C5F1_PCNA'
