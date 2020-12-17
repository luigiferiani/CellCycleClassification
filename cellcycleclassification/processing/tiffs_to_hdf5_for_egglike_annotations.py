#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:10:43 2020

@author: lferiani
"""

import cv2
import fire
import tables
import numpy as np
from pathlib import Path


TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)


def create_hdf5(images_dir: Path):
    """
    create_hdf5

    Convert a folder of tiffs
    into a dataset that works with the eggs annotator gui.

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

    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    # create output name
    output_hdf5 = (images_dir.parent
                   / (images_dir.stem + '.hdf5'))
    # get data
    figslist = list(images_dir.rglob('*.tiff'))
    import pdb
    pdb.set_trace()
    # load images. This can be bad if too many of them
    imgs = [cv2.imread(str(img_fname), -1)[None, :, :]
            for img_fname in figslist]
    imgs = np.concatenate(imgs, axis=0)
    # write data in output
    # and images
    with tables.File(output_hdf5, 'w') as fid:
        fid.create_earray('/',
                          'full_data',
                          obj=imgs,
                          filters=TABLE_FILTERS)

    return


def main():
    fire.Fire(create_hdf5)


# if __name__ == '__main__':
#     main()

if __name__ == '__main__':

    # where are things:
    work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    imgs_dir = work_dir / 'Mitotic_A549_nuclei'
    create_hdf5(imgs_dir)