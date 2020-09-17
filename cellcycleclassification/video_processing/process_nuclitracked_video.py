#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:38:03 2020

@author: lferiani
"""

import cv2
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch

from cellcycleclassification.processing.create_annotations_dataset import (
    read_nuclitrack_data, get_ordered_imgs_list)
from cellcycleclassification.classifier.scripts.eval import (
    model_path_to_name, get_training_parameters)
from cellcycleclassification.classifier.models.helper import (
    get_model_datasets_criterion)

PRED_COLOURS = {
    0: (200, 200, 200),  # grayish. this is BGR
    1: (0, 255, 255),   # green. this is BGR
    }


class IndexedFigsDir(object):

    def __init__(self, images_dir: Path):
        # sanitise input
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)
        figslist = get_ordered_imgs_list(images_dir)
        self.imgpath_df = pd.DataFrame(
            figslist,
            columns=['frame_number', 'filename']
            ).set_index('frame_number')
        self.imgpath_df['filename'] = self.imgpath_df['filename'].astype(str)

        self.frame_height, self.frame_width = self[0].shape[-2:]

    def __len__(self):
        return len(self.imgpath_df)

    def __getitem__(self, frame_number):
        if isinstance(frame_number, slice):
            warnings.warn(
                'Slice is used with pandas .loc, size mismatch is possible')
        fnames = self.imgpath_df.loc[frame_number, 'filename']
        if isinstance(fnames, str):
            fnames = [fnames]
        return np.concatenate([self._imread(f) for f in fnames], axis=0)

    def _imread(self, img_fname):
        im = cv2.imread(img_fname, -1)[None, :, :]
        return im



class VideoProcessor(object):

    def __init__(
            self,
            tracking_csv: Path,
            images_dir: Path,
            model_fname: Path):

        # sanitise input
        if isinstance(tracking_csv, str):
            tracking_csv = Path(tracking_csv)
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)

        # data
        self.tracking_csv_fname = tracking_csv
        self.images_dirname = images_dir
        self.tracks_df = read_nuclitrack_data(self.tracking_csv_fname)
        self.vid = IndexedFigsDir(images_dir=self.images_dirname)
        # data sizes
        # TODO: roi size is dictated by model
        self.roi_size = 48
        self.frame_height = self.vid.frame_height
        self.frame_width = self.vid.frame_width
        # prepare tracks_df for model
        self._calculate_roi_boundaries()
        # pytorch model things:
        self.model_fname = model_fname
        self.model_state_name = None
        self.model_name = None
        self.model = None
        self.device = None
        # initialise all pytorch things
        self._load_model()

    def _calculate_roi_boundaries(self):
        half_roi_sz = self.roi_size // 2
        # use a different dataframe to start with
        tracks = pd.DataFrame()
        # unbound values
        tracks['x_min'] = self.tracks_df['x_center'].astype(int) - half_roi_sz
        tracks['x_max'] = self.tracks_df['x_center'].astype(int) + half_roi_sz
        tracks['y_min'] = self.tracks_df['y_center'].astype(int) - half_roi_sz
        tracks['y_max'] = self.tracks_df['y_center'].astype(int) + half_roi_sz
        # how much out of bound?
        tracks['lpad'] = -np.minimum(tracks['x_min'], 0)
        tracks['rpad'] = np.maximum(tracks['x_max'] - self.frame_width, 0)
        tracks['tpad'] = -np.minimum(tracks['y_min'], 0)
        tracks['bpad'] = np.maximum(tracks['y_max'] - self.frame_height, 0)
        # constrain back to bounds.
        # clip is so much slower than np.minimum/maximum.
        # But readability wins and this is only done once
        tracks['x_min'] = tracks['x_min'].clip(lower=0)
        tracks['x_max'] = tracks['x_max'].clip(upper=self.frame_width)
        tracks['y_min'] = tracks['y_min'].clip(lower=0)
        tracks['y_max'] = tracks['y_max'].clip(upper=self.frame_height)
        # organise padding info in tuples already
        tracks['hpad'] = list(zip(tracks['lpad'], tracks['rpad']))
        tracks['vpad'] = list(zip(tracks['tpad'], tracks['bpad']))
        tracks['pad'] = list(zip(tracks['vpad'], tracks['hpad']))
        tracks = tracks.drop(
            columns=['lpad', 'rpad', 'tpad', 'bpad', 'vpad', 'hpad'])
        # write results back into self data
        self.tracks_df = self.tracks_df.join(tracks)
        return

    def get_rois_of_track_id(self, track_id):
        assert track_id in self.tracks_df['track_id'].values, (
            f'Unknown track id {track_id}')
        track_data = self.tracks_df.query(f'track_id == {track_id}')
        return self._get_rois_in_dataframe(track_data)

    def get_rois_in_frame(self, frame_number):
        frame_data = self.tracks_df.query(f'frame == {frame_number}')
        return self._get_rois_in_dataframe(frame_data)

    def _get_rois_in_dataframe(self, df, is_normalise=True):
        """
        df can be either a single frame dataframe, or anything, really.
        If one frame only, the image will be loaded ahead forr speedup
        """
        # pre-load for speedup
        if df['frame'].nunique() == 1:
            fn = df['frame'].values[0]
            img = self.vid[fn]
            assert img.shape[0] == 1
        else:
            img = None  # this will make _get_roi load the image every time
        # now loop on roi info, each pass gets one roi
        roi_data = defaultdict(np.array)
        for ind, info in df.iterrows():
            # get roi data
            this_roi = self._get_roi(info, img=img)
            # cast to float and apply normalisation
            if is_normalise:
                this_roi = self._img_normalise(this_roi.astype(np.float32))
            # write in accumulator
            roi_data[ind] = this_roi
        # convert to a stack
        roi_data = np.concatenate(
            [roi_data[k] for k in roi_data.keys()], axis=0)
        # add the channel dimension
        roi_data = roi_data[:, None, :, :]
        return roi_data

    def _get_roi(self, roi_info, img=None):
        if img is None:
            img = self.vid[roi_info['frame']]
        this_roi = img[
                :,
                roi_info['y_min']:roi_info['y_max'],
                roi_info['x_min']:roi_info['x_max']
                ]
        if np.any(roi_info['pad']):
            pad_with = np.percentile(this_roi, 5)
            padding = (0, 0), *roi_info['pad']
            this_roi = np.pad(
                this_roi,
                padding,
                mode='constant',
                constant_values=pad_with)
        assert this_roi.shape[-2:] == (self.roi_size, self.roi_size)
        return this_roi

    def _img_normalise(self, img):
        img -= img.mean()
        img /= img.std()
        return img

    def _load_model(self):
        if not self.model_fname:
            return
        # choose device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # get info from model path
        self.model_state_name, train_sess_name = model_path_to_name(
            self.model_fname)
        train_pars = get_training_parameters(train_sess_name)
        self.model_name = train_pars['model_name']
        self.model, _, _ = get_model_datasets_criterion(self.model_name)
        # load model state
        checkpoint = torch.load(self.model_fname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # set model in eval model straight away
        self.model.eval()
        return

    def process_video(self):
        assert self.model is not None
        # prepare column of right dtype for storing results
        self.tracks_df['predicted_id'] = -1
        with torch.no_grad():
            # loop on frames worth of rois
            for fn, frame_data in tqdm(self.tracks_df.groupby('frame')):
                frame_rois = self._get_rois_in_dataframe(frame_data)
                # make the rois a pytorch tensor
                frame_rois = torch.from_numpy(frame_rois).to(self.device)
                # evaluate
                out = self.model(frame_rois)
                if out.ndim > 1:
                    frame_predictions = torch.argmax(out, axis=1)
                else:
                    frame_predictions = (torch.sigmoid(out) > 0.5).long()
                self.tracks_df.loc[
                    frame_data.index, 'predicted_id'
                    ] = frame_predictions.cpu().numpy()

    def export_csv(self, out_path=None):
        if isinstance(out_path, str):
            out_path = Path(out_path)
        assert 'predicted_id' in self.tracks_df.columns, 'video not processed'
        if not out_path:
            out_path = (
                self.tracking_csv_fname.parent /
                (self.tracking_csv_fname.stem + '_classified')
                ).with_suffix('.csv')
        print(f'Saving to {out_path}')
        self.tracks_df.to_csv(out_path, index=False)
        return

    def export_frames(self, out_path=None):
        warnings.warn('These frames are just exported for quick visualisation')
        if isinstance(out_path, str):
            out_path = Path(out_path)
        assert 'predicted_id' in self.tracks_df.columns, 'video not processed'
        if not out_path:
            out_path = (
                self.images_dirname.parent /
                (self.images_dirname.name + '_classified')
                )
        print(f'Saving to {out_path}')
        out_path.mkdir(parents=True, exist_ok=True)
        # loop on frames
        for fn, frame_data in tqdm(self.tracks_df.groupby('frame')):
            # get an easy to manipulate image
            bgr_img = self._get_frame_as_bgr(fn)
            for _, roi_info in frame_data.iterrows():
                # cv2 works in (x,y)
                tl = roi_info['x_min'], roi_info['y_min']  # top left
                br = roi_info['x_max'], roi_info['y_max']  # bottom right
                color = PRED_COLOURS[roi_info['predicted_id']]
                bgr_img = cv2.rectangle(
                    bgr_img, tl, br, color, 2)
            # now write
            savename = str(out_path / f'frame_{fn:04d}.png')
            cv2.imwrite(savename, bgr_img)
        return

    def _get_frame_as_bgr(self, frame_number):
        # read raw frame as 2d array
        img = self.vid[frame_number].squeeze()
        # quick rescaling
        lower, upper = np.percentile(img, [10, 99.999])
        img = np.clip(img, lower, upper)
        img = img - lower
        img = img / (upper-lower)
        img = img * 255
        img = img.astype(np.uint8)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr_img



if __name__ == "__main__":
    work_dir = Path('/Users/lferiani/work_repos/CellCycleClassification/data')
    csv_fname = work_dir / 'R5C5F1_PCNA_sel.csv'
    imgs_dir = work_dir / 'R5C5F1_PCNA/'
    model_path = Path(
        '/Volumes/behavgenom$/Luigi/Data/AlexisBarr_cell_cycle_classification/'
        'trained_models')
    model_path /= 'v_04_60_20200908_160037/v_04_60_best.pth'

    vidproc = VideoProcessor(
        tracking_csv=csv_fname,
        images_dir=imgs_dir,
        model_fname='',
        )

    vidproc.process_video()
    vidproc.export_frames()

    # im = vidproc.frame[0]

    # ifd = IndexedFigsDir(imgs_dir)
    # img = ifd[0]


