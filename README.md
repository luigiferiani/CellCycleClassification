# CellCycleClassification

## Installation

1. Install [anaconda](https://www.anaconda.com/products/individual)

2. Using the terminal, clone the content of this repository:
```bash
mkdir ~/behavgenom_repos
cd ~/behavgenom_repos
git clone https://github.com/luigiferiani/CellCycleClassification.git
cd CellCycleClassification
```

3. Make sure that `conda-forge` is in the list of `conda` channels:
```bash
conda config --show-sources
```
If `conda-forge` does not appear in the list of channels, add it with
`conda config --add channels conda-forge`

4. Install the software in a dedicated `conda` environment:
```bash
conda env create --file environment.yml
conda activate cellcycleclassification
pip install -e .
```

## Update an existing installation

```bash
cd ~/behavgenom_repos/CellCycleClassification
git pull
conda env activate cellcycleclassification
conda env update -f environment.yml
pip install -e .
```

Tested on macOS.

## Use

### Classify NucliTrack-processed videos

#### To process a single file:
``` bash
conda activate cellcycleclassification
classify /path/to/nuclitracked_data_sel.csv
```

In this example, the program assumes the existence of a folder named
`/path/to/nuclitracked_data/` containing frames in tiff format.

#### To process all files in a folder:
``` bash
conda activate cellcycleclassification
classify /path/to/folder/with/nuclitracked/data
```

In this example, the program assumes the existence of csv files and folders
with the following naming convention:
```bash
/path/to/folder/with/nuclitracked/data/a_video_sel.csv        # NT csv
/path/to/folder/with/nuclitracked/data/a_video/               # frames folder
/path/to/folder/with/nuclitracked/data/another_video_sel.csv  # NT csv
/path/to/folder/with/nuclitracked/data/another_video/         # frames folder
/path/to/folder/with/nuclitracked/data/a_third_video_sel.csv  # NT csv
/path/to/folder/with/nuclitracked/data/a_third_video/         # frames folder
```

#### Output files
The results of the classification are exported to a csv file with name obtained
substituting `_sel.csv` with `_sel_classified.csv`.
E.g. `classify /path/to/folder/with/nuclitracked/data/a_video_sel.csv` will
output `/path/to/folder/with/nuclitracked/data/a_video_sel_classified.csv`

It is possible to export a series of annotated images by using the classify
command with the flag `--export_frames=True`.


### Create dataset for manual annotations

The first step is to create an annotation dataset starting from a video that was
processed with [NucliTrack](https://nuclitrack.readthedocs.io/en/latest/).

The video data is assumed to be in a folder of `uint16` tiffs, with the
frame number being written in the images' filename between the groups of letters
`sk` and `fk`.

To combine the video data and the `.csv` file created by NucliTrack into an
`_annotations.hdf5` file that can be then opened with the [GUI](#using-the-gui):
```bash
conda activate cellcycleclassification
create_dataset --images_dir path/to/the/tiffs/folder --tracking_csv path/to/nuclitrack/output.csv
```
The output dataset will be in the same folder as the NucliTrack `.csv`,
and have a name that shares the root with NucliTrack's csv and ends in
`_annotations.hdf5`.

### Manual annotations

#### Opening the GUI
```bash
conda activate cellcycleclassification
annotate_dataset
```

#### Loading a dataset
The dataset created at the previous step can now be opened either by using the
`Select File` button, or with a drag and drop into the empty field to its left.

The GUI supports resuming, so one can load a partially annotated dataset
and the the last annotated ROI will be shown
(to be precise, the last ROI in the dataset with an annotation,
not the most recently annotated ROI).

#### Using the GUI
While the GUI can be used with a mouse, the most common functionalities can be accessed via [keyboard shortcuts](#keyboard-shortcuts), for ease and speed.

The GUI shows the data on a track-by-track basis;
to move across frames of the current track, use either the slider
or the spinbox to its left.

To annotate a ROI, click on one of the buttons directly under the slider.
To undo an annotation, click the button again.

A ROI can be annotated as `No Cell` if there is no cell in the ROI
(this can happen since `create_dataset` interpolates gaps in the NucliTrack
results), or `Unsure` if it is not possible to determine the stage the cell
is in.
`Unsure` and `No Cell` ROIs will not be used for training CNNs.

`Prev Track` and `Next Track` will move across different tracks.
To select a particular track use the dropdown menu to the right of `Next Track`.

`frame-by-frame contrast` normalises each ROI's grey levels by only taking into
account the data in that ROI. This usually yields a more aggressive
contrast adjustment, but can fix occasional clipping of very bright cells.
When unticked, the contrast is adjusted by taking into account the grey levels
in the entire video.

`ROI only` crops the image tightly onto the cell in the current track.
When unticked, the cell is showed in a wider rectangle that encompasses
the entire space the cell roamed while being tracked.

`Show cell centre` paints an orange dot at the centre of the ROI, showing,
in case multiple cells are in the same ROI, which cell to focus on.

`Save` saves the progress on disk. Closing the GUI will also prompt you to save.

### Keyboard shortcuts

- `Left Key`: Move to the previous frame in the current track
- `Right Key`: Move to the next frame in the current track
- `,`, or `<`: Move to the previous track
- `.`, or `>`: Move to the next track
- `1`: Annotate the cell in the ROI as `G0`
- `2`: Annotate the cell in the ROI as `G1`
- `3`: Annotate the cell in the ROI as `S`
- `4`: Annotate the cell in the ROI as `G2`
- `5`: Annotate the cell in the ROI as `M`
- `6`: If there is no cell in the ROI
- `7`: If you are not sure about the label to give
