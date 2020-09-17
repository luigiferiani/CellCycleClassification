#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:37:29 2020

@author: lferiani
"""

import tqdm
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from cellcycleclassification.classifier.models.helper import get_dataloader


def train_one_epoch(
        basename, model, optimiser, criterion,
        data_loader, device,
        epoch, logger=None):
    # train on epoch
    model.train()
    # header = f'{basename} Train Epoch: [{epoch}]'

    train_loss = 0.0
    # pbar = tqdm.tqdm(data_loader, desc=header)
    # for mbc, data in enumerate(pbar):  # mini batches
    for mbc, data in enumerate(data_loader):  # mini batches
        # get the inputs; data is a list of [inputs, labels]
        batch_imgs, batch_labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimiser.zero_grad()
        # forward + backward + optimize
        out = model(batch_imgs)  # outputs logits or probas
        loss = criterion(out, batch_labels)
        loss.backward()
        optimiser.step()
        # store mini batch loss in accumulator
        train_loss += loss.item()
    # average train loss in this epoch
    train_loss /= mbc
    if logger:
        # store loss
        logger.add_scalar('train_epoch_loss', train_loss, epoch)

    return train_loss


@torch.no_grad()
def evaluate_one_epoch(
        basename, model, criterion, data_loader, device, epoch, logger=None,
        is_return_images=False, is_return_isfirstinstage=False):
    # evaluate after epoch
    model.eval()
    # header = f'{basename} Eval Epoch: [{epoch}]'

    val_loss = defaultdict(float)
    labels = defaultdict(torch.tensor)
    predictions = defaultdict(torch.tensor)
    if is_return_images:
        images = defaultdict(torch.tensor)
    else:
        images = None
    if is_return_isfirstinstage:
        isfirstinstage = defaultdict(torch.tensor)
    else:
        isfirstinstage = None
    # for mbc, data in enumerate(pbar):
    for mbc, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        batch_imgs, batch_labels = data[0].to(device), data[1].to(device)
        if is_return_isfirstinstage:
            batch_isfirstinstage = data[2]
        # forwards only
        out = model(batch_imgs)
        _loss = criterion(out, batch_labels)
        if out.ndim > 1:
            batch_predictions = torch.argmax(out, axis=1)
        else:
            batch_predictions = (torch.sigmoid(out) > 0.5).long()
        # store labels and predictions
        labels[mbc] = batch_labels.cpu()
        predictions[mbc] = batch_predictions.cpu()
        if is_return_images:
            images[mbc] = batch_imgs.cpu()
        if is_return_isfirstinstage:
            isfirstinstage[mbc] = batch_isfirstinstage
        # store mini batch loss in accumulator
        val_loss[mbc] = _loss.item()

    # average
    val_loss = np.mean([val_loss[mbc] for mbc in val_loss.keys()])

    # concatenate accumulators into np arrays for ease of use.
    def _numpify(od):
        return np.concatenate([od[k].squeeze() for k in od.keys()], axis=0)

    predictions = _numpify(predictions)
    labels = _numpify(labels).astype(int)  # some loss criteria needed a float

    if is_return_images:
        images = _numpify(images)
    if is_return_isfirstinstage:
        isfirstinstage = _numpify(isfirstinstage)
    # measures
    class_rep = classification_report(labels, predictions, output_dict=True)
    val_accuracy = class_rep['accuracy']
    # store loss
    if logger:  # if I just want to use this to evaluate a model
        logger.add_scalar('val_epoch_loss', val_loss, epoch)
        logger.add_scalar('accuracy', val_accuracy, epoch)
        logger.add_scalar('precision', class_rep['1']['precision'], epoch)
        logger.add_scalar('recall', class_rep['1']['recall'], epoch)
        logger.add_scalar('specificity', class_rep['0']['recall'], epoch)
        logger.add_scalar('f1-score', class_rep['1']['f1-score'], epoch)

    return val_loss, val_accuracy, predictions, labels, images, isfirstinstage


def train_model(
        save_prefix,
        model,
        device,
        train_dataset,
        val_dataset,
        criterion,
        optimiser,
        log_dir,
        scheduler=None,
        n_epochs=100,
        batch_size=64,
        num_workers=4,
        is_use_sampler=False,
        ):

    # add datatime to save_prefix
    strnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_prefix = f'{save_prefix}_{strnow}'

    # create dataloaders
    train_loader = get_dataloader(
        train_dataset, is_use_sampler, batch_size, num_workers)
    val_loader = get_dataloader(
        val_dataset, is_use_sampler, batch_size, num_workers)

    # get logger ready
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir=str(log_dir))
    print(f'Logging at {log_dir}')

    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs, desc=save_prefix)
    for epoch in pbar_epoch:
        train_one_epoch(
            save_prefix,
            model,
            optimiser,
            criterion,
            train_loader,
            device,
            epoch,
            logger,
        )
        val_loss, val_accuracy, _, _, _, _ = evaluate_one_epoch(
            save_prefix,
            model,
            criterion,
            val_loader,
            device,
            epoch,
            logger,
        )

        # update scheduler
        if scheduler:
            if scheduler.mode == 'min':
                scheduler.step(val_loss)
            elif scheduler.mode == 'max':
                scheduler.step(val_accuracy)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        training_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
        }
        if scheduler:
            training_state['scheduler_state_dict'] = scheduler.state_dict()

        if is_best:
            torch.save(
                training_state,
                log_dir/save_prefix.replace(strnow, 'best.pth')
                )

    print('Finished training')
    model_savepath = log_dir / f'{save_prefix}.pth'
    torch.save(
        training_state, model_savepath)
    print(f'model saved at {model_savepath}')
    logger.flush()
    logger.close()
