#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:10:31 2021

@author: lferiani
"""

import fire
from tqdm import tqdm
from pathlib import Path

from cellcycleclassification import BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH
from cellcycleclassification.video_processing.Processor import (
    DualModelVideoProcessor
)


def csvfname2imgdir(csv_fname: Path):
    if isinstance(csv_fname, str):
        csv_fname = Path(csv_fname)
    return csv_fname.parent / csv_fname.stem.replace('_sel', '')


def process_directory(working_dir: Path, is_export_frames: bool = False):
    """
    Process the nuclitrack files in the input_path.
    On each video run inference with two pytorch models,
    combine the outputs into a single prediction,
    and export a csv with the predicted labels.

    Parameters
    ----------
    working_dir : Path
        Path to a folder containing nuclitrack csvs ad image folders,
        assuming this type of naming convention:
        input_path/random_video_name_sel.csv  -> nuclitrack csv
        input_path/random_video_name/         -> folder of frames
        input_path/another_video_name_sel.csv -> nuclitrack csv
        input_path/another_video_name/        -> folder of frames

    is_export_frames: Bool, default False
        If True, export a preview of classified cells.

    Returns
    -------
    None.

    """
    # look for suitable files, with existing image directory
    csv_fnames = [csv_fname
                  for csv_fname in working_dir.rglob('*_sel.csv')
                  if csvfname2imgdir(csv_fname).exists()]
    print(f'Found {len(csv_fnames)} videos to classify.', flush=True)

    for csv_fname in tqdm(csv_fnames, desc='Processing video '):
        process_one(csv_fname, is_export_frames=is_export_frames)


def process_one(tracking_csv: Path, is_export_frames: bool = False):
    """
    Process the tracking_csv by running inference with two pytorch models,
    combining the outputs into a single prediction,
    and exporting a csv with the predicted labels.

    Parameters
    ----------
    tracking_csv : Path
        Path to a nuclitrack csv
        This file structure is assumed:
        /path/to/video_data_sel.csv -> the nuclitrack csv
        /path/to/video_data/        -> folder containing video frames as tiffs

    is_export_frames: Bool, default True
        If True, export a preview of classified cells.

    Returns
    -------
    None.

    """

    # input checks
    assert tracking_csv.name.endswith('_sel.csv'), (
        "Expecting a csv file ending in `_sel.csv`")
    image_dir = csvfname2imgdir(tracking_csv)
    assert image_dir.exists(), f"Cannot find the data folder at {image_dir}"

    vidproc = DualModelVideoProcessor(
        tracking_csv=tracking_csv,
        images_dir=image_dir,
        binary_model_fname=BINARY_MODEL_PATH,
        multiclass_model_fname=MULTICLASS_MODEL_PATH,
        )
    vidproc.process_video()
    vidproc.export_csv()
    if is_export_frames:
        vidproc.export_frames()

    return


def process(input_path: Path, export_frames: bool = False):
    """
    Process the input_path (or the files in the input_path)
    by running inference with two pytorch models,
    combining the outputs into a single prediction,
    and exporting a csv with the predicted labels.

    Parameters
    ----------
    input_path : Path
        Path to a nuclitrack csv or a folder containing multiple
        nuclitrack csvs.
        If input_path is a single csv file, this file structure is assumed:
        /path/to/video_data_sel.csv -> the nuclitrack csv
        /path/to/video_data/        -> folder containing video frames as tiffs
        If input_path is a folder, its content is assumed to be of the type:
        input_path/random_video_name_sel.csv  -> nuclitrack csv
        input_path/random_video_name/         -> folder of frames
        input_path/another_video_name_sel.csv -> nuclitrack csv
        input_path/another_video_name/        -> folder of frames

    export_frames: Bool, default False
        If True, export a preview of classified cells.

    Returns
    -------
    None.

    """
    # sanitise input
    if isinstance(input_path, str):
        input_path = Path(input_path)
    #   send to appropriate processing function
    if input_path.is_dir():
        process_directory(input_path, is_export_frames=export_frames)
    else:
        process_one(input_path, is_export_frames=export_frames)


def main():
    fire.Fire(process)


if __name__ == '__main__':
    main()
