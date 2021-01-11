#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:38:03 2020

@author: lferiani
"""
# %%

import cv2
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

from cellcycleclassification.processing.create_annotations_dataset import (
    read_nuclitrack_data, get_ordered_imgs_list)
from cellcycleclassification.classifier.scripts.eval import (
    model_path_to_name)
from cellcycleclassification.classifier.utils import get_training_parameters
from cellcycleclassification.classifier.models.helper import (
    get_model_datasets_criterion)
from cellcycleclassification.video_processing.utils import (
    despike, find_forbidden_transitions)

PRED_COLOURS = {
    -1: (0, 0, 0),
    0: (200, 200, 200),  # grayish. this is BGR
    1: (0, 255, 255),   # yellow. this is BGR
    2: (255, 255, 0),   # cyan. this is BGR
    3: (0, 0, 255),     # red
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


class VideoProcessorBase(object):
    """
    Dummy class for inheritance purposes.
    Does not do any inference
    """
    def __init__(
            self,
            tracking_csv: Path,
            images_dir: Path):

        # sanitise input
        if isinstance(tracking_csv, str):
            tracking_csv = Path(tracking_csv)
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)

        # data
        self.tracking_csv_fname = tracking_csv
        self.images_dir = images_dir
        self.tracks_df = None
        self.vid = IndexedFigsDir(images_dir=self.images_dir)
        self.frame_height = self.vid.frame_height
        self.frame_width = self.vid.frame_width
        self.roi_size = 48
        # load
        self._load_nuclitrack_data()

    def _load_nuclitrack_data(self):
        """
        Wrapper for read_nuclitrack data,
        gets rid of extra columns, fixes dtype
        """
        tracks_df = read_nuclitrack_data(self.tracking_csv_fname)
        tracks_df = tracks_df.drop(columns='label_id')
        tracks_df['track_id'] = tracks_df['track_id'].astype(int)
        self.tracks_df = tracks_df
        return

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

    def process_video(self):
        """
        Dummy function that creates a column full of -1
        """
        self.tracks_df['predicted_id'] = -1

    def export_csv(self, out_path=None):
        if isinstance(out_path, str):
            out_path = Path(out_path)
        assert 'predicted_id' in self.tracks_df.columns, 'video not processed'
        if not out_path:
            out_path = (
                self.tracking_csv_fname.parent /
                (self.tracking_csv_fname.stem + '_classified')
                ).with_suffix('.csv')
        print(f'Saving to {out_path}', flush=True)
        self.tracks_df.to_csv(out_path, index=False)
        return

    def export_frames(self, out_path=None):
        warnings.warn('These frames are just exported for quick visualisation')
        if isinstance(out_path, str):
            out_path = Path(out_path)
        assert 'predicted_id' in self.tracks_df.columns, 'video not processed'
        if not out_path:
            out_path = (
                self.images_dir.parent /
                (self.images_dir.name + '_classified')
                )
        print(f'Saving to {out_path}', flush=True)
        out_path.mkdir(parents=True, exist_ok=True)
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        font_thickness = 1
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
                bgr_img = cv2.putText(
                    bgr_img, str(roi_info['track_id']), tl,
                    font, font_scale, color, font_thickness, cv2.LINE_AA)
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


class VideoProcessor(VideoProcessorBase):

    def __init__(
            self,
            tracking_csv: Path,
            images_dir: Path,
            model_fname: Path):

        super().__init__(tracking_csv, images_dir)

        # pytorch model things:
        self.model_fname = model_fname
        self.model_state_name = None
        self.model_name = None
        self.model = None
        self.device = None
        self.probas = None
        # initialise
        self._load_model()
        self._calculate_roi_boundaries()

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
        self.roi_size = train_pars['roi_size']
        if self.roi_size is None:
            self.roi_size = 48  # default value
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
                    frame_probas = F.softmax(out, dim=1)
                else:
                    frame_probas = torch.sigmoid(out)
                    frame_predictions = (frame_probas > 0.5).long()
                # store predictions
                self.tracks_df.loc[
                    frame_data.index, 'predicted_id'
                    ] = frame_predictions.cpu().numpy()
                # store probabilities too
                if self.probas is None:
                    # initialise self.probas
                    if frame_probas.ndim > 1:
                        sz = (len(self.tracks_df), frame_probas.shape[1])
                    else:
                        sz = (len(self.tracks_df),)
                    self.probas = np.zeros(sz)
                # store probas in self.probas
                self.probas[frame_data.index] = frame_probas.cpu().numpy()
        return


class DualModelVideoProcessor(VideoProcessorBase):

    def __init__(
            self,
            tracking_csv: Path,
            images_dir: Path,
            binary_model_fname: Path,
            multiclass_model_fname: Path,
            ):

        super().__init__(tracking_csv, images_dir)

        self.bin_model_fname = binary_model_fname
        self.multi_model_fname = multiclass_model_fname
        # initialise the two single-model processors
        self.bvp = VideoProcessor(
            self.tracking_csv_fname, self.images_dir, self.bin_model_fname)
        self.mvp = VideoProcessor(
            self.tracking_csv_fname, self.images_dir, self.multi_model_fname)

        # get right roi size and update tracks_df
        self.roi_size = min(self.mvp.roi_size, self.bvp.roi_size)
        self._calculate_roi_boundaries()

        # initialise combined data fields:
        self.combined_probas = None
        self.labels_dict = {0: 'G0/1', 1: 'S', 2: 'G2', 3: 'M'}

    def run_inference(self):
        # run inference in the two models
        print('Running binary classifier...', flush=True)
        self.bvp.process_video()
        print(flush=True)
        print('Running multiclass classifier...', flush=True)
        self.mvp.process_video()
        # combine the probabilities. bin probas is prob of class 1
        n_labels = self.mvp.probas.shape[1]
        bin_probas_expanded = np.array(
            [(1-self.bvp.probas)/(n_labels-1), self.bvp.probas])
        bin_probas_expanded = bin_probas_expanded[[0, 1, 0, 0], :].T
        assert bin_probas_expanded.shape[1] == n_labels
        self.combined_probas = bin_probas_expanded/2 + self.mvp.probas/2
        # combine the tracks_df
        self.tracks_df['predicted_id_raw'] = np.argmax(
            self.combined_probas, axis=1)

    def postprocess(self):
        # remove single-frame spikes
        processed_df = despike(
            self.tracks_df, col='predicted_id_raw', col_post='predicted_id')
        # find all dubious transitions
        self.tracks_df = find_forbidden_transitions(
            processed_df, col='predicted_id', col_forbidden='is_forbidden')
        self.tracks_df['predicted_label'] = self.tracks_df['predicted_id'].map(
            self.labels_dict)
        return

    def process_video(self):
        self.run_inference()
        self.postprocess()
        return


# %%
if __name__ == "__main__":
    from cellcycleclassification import (
        BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH)

    work_dir = Path('/Users/lferiani/work_repos/CellCycleClassification/data')
    csv_fname = work_dir / 'R5C5F1_PCNA_sel.csv'
    imgs_dir = work_dir / 'R5C5F1_PCNA/'
    models_dir = Path(
        '/Volumes/behavgenom$/Luigi/Data/AlexisBarr_cell_cycle_classification/'
        'trained_models')
    # model_path /= 'v_04_60_20200908_160037/v_04_60_best.pth'
    # bin_model_path = (
    #     models_dir / 'v_06_60_20201217_113641/v_06_60_best.pth')
    # multi_model_path = (
    #     models_dir / 'v_12_63_20201218_213041/v_12_63_20201218_213041.pth')
    bin_model_path = BINARY_MODEL_PATH
    multi_model_path = MULTICLASS_MODEL_PATH

    vidproc = DualModelVideoProcessor(
        tracking_csv=csv_fname,
        images_dir=imgs_dir,
        binary_model_fname=bin_model_path,
        multiclass_model_fname=multi_model_path,
        )
    vidproc.process_video()
    vidproc.export_csv()
    vidproc.export_frames()
