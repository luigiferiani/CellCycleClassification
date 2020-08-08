#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:15:48 2020

@author: lferiani
"""

import sys
import tables
import pandas as pd
from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, qRgb
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QComboBox, QHBoxLayout, QMessageBox)

from HDF5VideoPlayer import HDF5VideoPlayerGUI, LineEditDragDrop

INT_COLS = ['track_id', 'frame']

BUTTON_STYLESHEET_STR = (
    "QPushButton:checked "
    + "{border: 2px solid; border-radius: 6px; background-color: %s }"
    )

BTN_COLOURS = {1: 'green',
               2: 'orange',
               3: 'yellow',
               4: 'brown',
               5: 'orange',
               6: 'darkRed',
               7: 'magenta',
               }


def _updateUI(ui):

    # delete things
    ui.horizontalLayout_2.addWidget(ui.playButton)
    ui.playButton.deleteLater()
    ui.playButton = None

    ui.horizontalLayout_6.removeWidget(ui.pushButton_h5groups)
    ui.pushButton_h5groups.deleteLater()
    ui.pushButton_h5groups = None

    ui.horizontalLayout_3.removeWidget(ui.doubleSpinBox_fps)
    ui.doubleSpinBox_fps.deleteLater()
    ui.doubleSpinBox_fps = None

    ui.horizontalLayout_3.removeWidget(ui.label_frame)
    ui.label_frame.deleteLater()
    ui.label_frame = None

    ui.horizontalLayout_3.removeWidget(ui.label_fps)
    ui.label_fps.deleteLater()
    ui.label_fps = None

    ui.horizontalLayout_3.removeWidget(ui.spinBox_step)
    ui.spinBox_step.deleteLater()
    ui.spinBox_step = None

    ui.horizontalLayout_3.removeWidget(ui.label_step)
    ui.label_step.deleteLater()
    ui.label_step = None

    ui.horizontalLayout_6.removeWidget(ui.comboBox_h5path)
    ui.comboBox_h5path.deleteLater()
    ui.comboBox_h5path = None

    # Remove all layouts
    ui.horizontalLayout.deleteLater()
    ui.horizontalLayout = None
    ui.horizontalLayout_2.deleteLater()
    ui.horizontalLayout_2 = None
    ui.horizontalLayout_3.deleteLater()
    ui.horizontalLayout_3 = None
    ui.horizontalLayout_6.deleteLater()
    ui.horizontalLayout_6 = None

    # define layouts
    ui.horizontalLayout = QHBoxLayout()
    ui.horizontalLayout.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout.setSpacing(6)
    ui.horizontalLayout.setObjectName("horizontalLayout")
    ui.verticalLayout.addLayout(ui.horizontalLayout)

    ui.horizontalLayout_2 = QHBoxLayout()
    ui.horizontalLayout_2.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout_2.setSpacing(6)
    ui.horizontalLayout_2.setObjectName("horizontalLayout_2")
    ui.verticalLayout.addLayout(ui.horizontalLayout_2)

    ui.horizontalLayout_3 = QHBoxLayout()
    ui.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout_3.setSpacing(6)
    ui.horizontalLayout_3.setObjectName("horizontalLayout_3")
    ui.verticalLayout.addLayout(ui.horizontalLayout_3)

    ui.horizontalLayout_4 = QHBoxLayout()
    ui.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout_4.setSpacing(6)
    ui.horizontalLayout_4.setObjectName("horizontalLayout_4")
    ui.verticalLayout.addLayout(ui.horizontalLayout_4)

    ui.horizontalLayout_5 = QHBoxLayout()
    ui.horizontalLayout_5.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout_5.setSpacing(6)
    ui.horizontalLayout_5.setObjectName("horizontalLayout_5")
    ui.verticalLayout.addLayout(ui.horizontalLayout_5)

    ui.horizontalLayout_6 = QHBoxLayout()
    ui.horizontalLayout_6.setContentsMargins(11, 11, 11, 11)
    ui.horizontalLayout_6.setSpacing(6)
    ui.horizontalLayout_6.setObjectName("horizontalLayout_6")
    ui.verticalLayout.addLayout(ui.horizontalLayout_6)

    # place widgets:
    # first layer
    ui.horizontalLayout_2.removeWidget(ui.spinBox_frame)
    ui.horizontalLayout.addWidget(ui.spinBox_frame)

    ui.horizontalLayout_2.removeWidget(ui.imageSlider)
    ui.horizontalLayout.addWidget(ui.imageSlider)

    # second layer
    ui.stage_g0_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.stage_g0_b)
    ui.stage_g0_b.setText("[1] G0")

    ui.stage_g1_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.stage_g1_b)
    ui.stage_g1_b.setText("[2] G1")

    ui.stage_s_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.stage_s_b)
    ui.stage_s_b.setText("[3] S")

    ui.stage_g2_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.stage_g2_b)
    ui.stage_g2_b.setText("[4] G2")

    ui.stage_m_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.stage_m_b)
    ui.stage_m_b.setText("[5] M")

    ui.no_cell_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.no_cell_b)
    ui.no_cell_b.setText("[6] No Cell")

    ui.unsure_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_2.addWidget(ui.unsure_b)
    ui.unsure_b.setText("[7] Unsure")

    # third layer
    ui.prev_track_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_3.addWidget(ui.prev_track_b)
    ui.prev_track_b.setText("Prev Track")

    ui.next_track_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_3.addWidget(ui.next_track_b)
    ui.next_track_b.setText("Next Track")

    ui.tracks_comboBox = QComboBox(ui.centralWidget)
    ui.tracks_comboBox.setEditable(False)
    ui.tracks_comboBox.setObjectName("tracks_comboBox")
    ui.tracks_comboBox.addItem("")
    ui.horizontalLayout_3.addWidget(ui.tracks_comboBox)

    ui.label_track_id = QLabel(ui.centralWidget)
    ui.label_track_id.setObjectName("label_track_id")
    ui.label_track_id.setText("track_id: ")
    ui.horizontalLayout_3.addWidget(ui.label_track_id)

    ui.label_track_counter = QLabel(ui.centralWidget)
    ui.label_track_counter.setObjectName("label_track_counter")
    ui.label_track_counter.setText("#/##")
    ui.horizontalLayout_3.addWidget(ui.label_track_counter)

    # fourth layer
    ui.save_b = QPushButton(ui.centralWidget)
    ui.horizontalLayout_4.addWidget(ui.save_b)
    ui.save_b.setText("Save")

    # fifth layer
    ui.label_dataset = QLabel(ui.centralWidget)
    ui.label_dataset.setObjectName("label_dataset")
    ui.label_dataset.setText("dataset name")
    ui.horizontalLayout_5.addWidget(ui.label_dataset)

    # ui.horizontalLayout.removeWidget(ui.lineEdit_video)
    ui.horizontalLayout_6.addWidget(ui.lineEdit_video)

    # ui.horizontalLayout.removeWidget(ui.pushButton_video)
    ui.horizontalLayout_6.addWidget(ui.pushButton_video)

    return ui


def check_good_input(fname):
    is_good = False
    is_hdf5 = str(fname).endswith('_annotations.hdf5')
    if is_hdf5:
        with tables.File(fname, 'r') as fid:
            is_good = all(df in fid
                          for df in ['full_data', '/annotations_df'])
    return is_good


class CellCycleAnnotator(HDF5VideoPlayerGUI):

    def __init__(self, ui=None):

        super().__init__()

        # Set up the user interface
        self.ui = _updateUI(self.ui)

        self.track_id = None
        self.track_counter = None  # from 0 to len(track_ids)
        self.track_ids = []
        self.annotations_df = None

        self.roi_size = 80
        self.buttons = {1: self.ui.stage_g0_b,
                        2: self.ui.stage_g1_b,
                        3: self.ui.stage_s_b,
                        4: self.ui.stage_g2_b,
                        5: self.ui.stage_m_b,
                        6: self.ui.no_cell_b,
                        7: self.ui.unsure_b}

        self.frame_number = 0
        # note: the ROI to show is defined by fram_number and track_id

        self.min_frame = 0
        self.tot_frames = 50
        self.ui.spinBox_frame.setMaximum(self.tot_frames - 1)
        self.ui.imageSlider.setMaximum(self.tot_frames - 1)
        # also initialise limits for the track
        self.track_limits = []

        # connect ui elements to functions
        LineEditDragDrop(
            self.ui.lineEdit_video,
            self.updateDataset,
            check_good_input)

        self.ui.tracks_comboBox.activated.connect(self.updateImGroup)
        self.ui.tracks_comboBox.currentIndexChanged.connect(self.updateImGroup)
        self.ui.next_track_b.clicked.connect(self.nextTrack_fun)
        self.ui.prev_track_b.clicked.connect(self.prevTrack_fun)
        self.ui.save_b.clicked.connect(self.save_to_disk_fun)
        self._setup_buttons()

        self.updateVideoFile = self.updateDataset  # alias

        return

    @property
    def frame_step(self):
        return 1

    def updateDataset(self, dataset_fname):

        # close the if there was another file opened before.
        if self.fid is not None:
            self.fid.close()
            self.mainImage.cleanCanvas()
            self.fid = None
            self.image_group = None
            self.track_id = None
            self.track_ids = []

            self.ui.tracks_comboBox.clear()
            self.annotations_df = None

        # save the name of the file and display it on the gui
        self.dataset_fname = dataset_fname
        self.ui.label_dataset.setText(self.dataset_fname)
        # read the big img stack in memory (can be changed in the future)
        with tables.File(dataset_fname, 'r') as fid:
            self.image_group = fid.get_node('/full_data').read()
        # adjust parameters for showing
        self.tot_frames = self.image_group.shape[0]
        self.image_height = self.image_group.shape[1]
        self.image_width = self.image_group.shape[2]
        # also initialise better limits for the track
        self.track_limits = [0, 0, self.image_height, self.image_width]

        # also read annotations
        self.annotations_df = pd.read_hdf(self.dataset_fname,
                                          key='/annotations_df')
        for col in INT_COLS:
            self.annotations_df[col] = self.annotations_df[col].astype(int)

        # get tracks list
        self.track_ids = list(self.annotations_df['track_id'].unique())

        # set index so it's easier to traverse the dataframe
        self.annotations_df.set_index(keys=['track_id', 'frame'], inplace=True)

        # set up combobox
        self.ui.tracks_comboBox.clear()
        for track_counter, track_id in enumerate(self.track_ids):
            self.ui.tracks_comboBox.addItem(str(track_id))

        # get the data related to the first track to show.
        # should be the first unlabelled one
        track_counter, frame_number = self.find_first_unlabelled_ROI()
        self.track_id = self.track_ids[track_counter]

        self.updateImGroup(track_counter, at_frame=frame_number)

        return

    def find_first_unlabelled_ROI(self):
        idx = self.annotations_df['label_id'] == 0
        (tid, fn) = self.annotations_df[idx].iloc[0].name
        tc = self.ui.tracks_comboBox.findText(str(tid))  # get track counter
        return tc, fn

    def updateImGroup(self, track_counter, at_frame=None):
        # this selects the right range of slider and updates track_id
        if track_counter < 0:
            # this happens when clearing the combobox
            return

        self.track_id = int(self.ui.tracks_comboBox.itemText(track_counter))
        self.track_counter = track_counter
        self.ui.label_track_id.setText(f'track_id: {self.track_id}')
        self.ui.label_track_counter.setText(
            (f'{self.track_counter+1}/'
             + f'{len(self.track_ids)}'))

        # find limits to track
        self.find_tracks_limits()

        # adjust limits to slider
        min_frame, max_frame = self.annotations_df.loc[
            self.track_id].index[[0, -1]]

        self.ui.spinBox_frame.setMinimum(min_frame)
        self.ui.spinBox_frame.setMaximum(max_frame)
        self.ui.imageSlider.setMinimum(min_frame)
        self.ui.imageSlider.setMaximum(max_frame)

        if at_frame is None:
            # select first frame of this track
            self.frame_number = min_frame
        else:
            # or the first un-labelled if resuming
            self.frame_number = at_frame

        self.ui.spinBox_frame.setValue(self.frame_number)
        self.updateImage()
        self.mainImage.zoomFitInView()

        return

    def find_tracks_limits(self):

        limits = self.annotations_df.loc[
            self.track_id, ['x_center', 'y_center']].agg([min, max])

        min_row = int(round(limits.loc['min', 'y_center'] - self.roi_size/2))
        max_row = int(round(limits.loc['max', 'y_center'] + self.roi_size/2))
        min_col = int(round(limits.loc['min', 'x_center'] - self.roi_size/2))
        max_col = int(round(limits.loc['max', 'x_center'] + self.roi_size/2))

        self.track_limits = [min_row, max_row, min_col, max_col]

        return

    def updateImage(self):
        self.readCurrentFrame()
        self.drawROIBoundaries()
        self.mainImage.setPixmap(self.frame_qimg)
        self._refresh_buttons()

    def readCurrentFrame(self):
        if self.image_group is None:
            self.frame_qimg = None
            return
        self.readCurrentROI()
        self._normalizeImage()

    def readCurrentROI(self):
        # row = self.annotations_df.loc[(self.track_id, self.frame_number)]
        # xc = int(round(row['x_center']))
        # yc = int(round(row['y_center']))
        # roi_min_col = max(0, xc - self.roi_size // 2)
        # roi_max_col = min(self.image_width, xc + self.roi_size // 2)
        # roi_min_row = max(0, yc - self.roi_size // 2)
        # roi_max_row = min(self.image_height, yc + self.roi_size // 2)
        # self.frame_img = self.image_group[self.frame_number,
        #                                   roi_min_row:roi_max_row,
        #                                   roi_min_col:roi_max_col]

        [min_row, max_row, min_col, max_col] = self.track_limits
        self.frame_img = self.image_group[self.frame_number,
                                          min_row:max_row,
                                          min_col:max_col]

    def drawROIBoundaries(self):

        # find left, bottom, width, height of box to draw
        row = self.annotations_df.loc[(self.track_id, self.frame_number)]
        xc = int(round(row['x_center']))
        yc = int(round(row['y_center']))
        # apply offset because FOV is cropped
        xc = int(xc - self.track_limits[2])
        yc = int(yc - self.track_limits[0])

        rect_specs = [xc - self.roi_size // 2,
                      yc - self.roi_size // 2,
                      self.roi_size,
                      self.roi_size]

        # draw
        painter = QPainter()
        painter.begin(self.frame_qimg)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(250, 140, 0))
        painter.setPen(pen)
        painter.drawRect(*rect_specs)
        painter.drawPoint(xc, yc)

    def nextTrack_fun(self):
        self.ui.tracks_comboBox.setCurrentIndex(
            min(self.track_counter + 1,
                self.ui.tracks_comboBox.count()-1))
        return

    def prevTrack_fun(self):
        self.ui.tracks_comboBox.setCurrentIndex(
            max(0, self.track_counter - 1))
        return

    def _setup_buttons(self):
        """
        Control button appearance and behaviour
        """
        stylesheet_str = BUTTON_STYLESHEET_STR

        # function called when a button is activated
        def _make_label(label_id, checked):
            """
            if function called when checking a button,
            loop through all the other buttons and uncheck those.
            And set well's label to be this checked button.'
            If unchecking a checked button, delete the existing annotation"""
            if checked:
                for btn_id, btn in self.buttons.items():
                    if btn_id != label_id:
                        btn.setChecked(False)
                    btn.repaint()

            if self.annotations_df is not None:
                # find well index
                if checked:
                    # add label
                    self.annotations_df.loc[
                        (self.track_id, self.frame_number),
                        'label_id'] = label_id
                else:
                    old_lab = self.annotations_df.loc[
                        (self.track_id, self.frame_number), 'label_id']
                    if old_lab == label_id:
                        # if the labeld was unchecked remove the label
                        self.annotations_df.loc[
                            (self.track_id, self.frame_number), 'label_id'] = 0
        # connect ui elements to callback function
        for btn_id, btn in self.buttons.items():
            btn.setCheckable(True)
            btn.setStyleSheet(stylesheet_str % BTN_COLOURS[btn_id])
            btn.toggled.connect(partial(_make_label, btn_id))
        return

    def _refresh_buttons(self,):
        # get current label:
        label_id = self.annotations_df.loc[
            (self.track_id, self.frame_number), 'label_id']
        if label_id > 0:
            self.buttons[label_id].setChecked(True)
        else:
            for btn in self.buttons.values():
                btn.setChecked(False)
                btn.repaint()

    def keyPressEvent(self, event):
        # HOT KEYS
        key = event.key()

        # Move to next track when pressed:  > or .
        if key == Qt.Key_Greater or key == Qt.Key_Period:
            self.nextTrack_fun()

        # Move to next track when pressed: < or ,
        elif key == Qt.Key_Less or key == Qt.Key_Comma:
            self.prevTrack_fun()

        # Move backwards when left arrow is
        elif key == Qt.Key_Left:
            self.frame_number -= self.frame_step
            if self.frame_number < self.ui.spinBox_frame.minimum():
                self.frame_number = self.ui.spinBox_frame.minimum()
            self.ui.spinBox_frame.setValue(self.frame_number)

        # Move to next frame when right arrow is pressed
        elif key == Qt.Key_Right:
            self.frame_number += self.frame_step
            if self.frame_number > self.ui.spinBox_frame.maximum():
                self.frame_number = self.ui.spinBox_frame.maximum()
            self.ui.spinBox_frame.setValue(self.frame_number)

        else:
            for btn_id, btn in self.buttons.items():
                if key == (Qt.Key_0+btn_id):
                    btn.toggle()
                    return
        return

    def save_to_disk_fun(self):
        self.annotations_df.reset_index(drop=False).to_hdf(
            self.dataset_fname,
            key='/annotations_df',
            index=False,
            mode='r+')
        return

    def closeEvent(self, event):
        quit_msg = "Do you want to save the current progress before exiting?"
        reply = QMessageBox.question(
            self,
            'Message',
            quit_msg,
            QMessageBox.No | QMessageBox.Yes,
            QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            self.save_to_disk_fun()

        super().closeEvent(event)
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = CellCycleAnnotator()
    ui.show()

    sys.exit(app.exec_())

