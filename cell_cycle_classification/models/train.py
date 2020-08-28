#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:37:29 2020

@author: lferiani
"""
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from models import CNN_tierpsy
from datasets import CellsDataset



if __name__ == "__main__":

    # where are things?
    work_dir = Path('~/work_repos/CellCycleClassification/data').expanduser()
    dataset_fname = work_dir / 'R5C5F1_PCNA_sel_annotations.hdf5'
    # old_savestate_fname = work_dir / 'CNN_tierpsy_state_20200828_171853.pth'
    old_savestate_fname = None

    # parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device in use: {device}')
    batch_size = 64
    # learning_rate = 1e-3
    learning_rate = 1e-4
    n_epochs = 3  # complete passes

    # create datasets
    train_data = CellsDataset(dataset_fname, which_set='train')
    val_data = CellsDataset(dataset_fname, which_set='val')
    test_data = CellsDataset(dataset_fname, which_set='test')

    # create dataloaders
    # num_workers=4 crashes in my spyder but works on pure python
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        test_data, shuffle=True, batch_size=batch_size, num_workers=4)

    # instantiate the model and optimizer
    model = CNN_tierpsy().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # in case there's a pre-trained bit:
    if old_savestate_fname:
        checkpoint = torch.load(old_savestate_fname, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
        model = model.to(device)
    else:
        last_epoch = -1

    # loop over the dataset multiple times
    epoch = last_epoch+1
    for epoch in range(epoch, epoch+n_epochs):

        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            print_every = 10  # print every 10 mini-batches
            running_loss += loss.item()  # running mean accumulator
            if i % print_every == print_every-1:
                print(f'[{epoch}, {i:04d}] loss: {running_loss/10}')
                running_loss = 0.0

    # save state
    strnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_savename = work_dir / f'CNN_tierpsy_state_{strnow}.pth'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss': loss,
            }, model_savename)


# print('Finished Training')
