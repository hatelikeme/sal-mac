import os
import shutil
import argparse
import math

from losses import ContrastiveLoss, TripletLoss

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable

def parse_parameters(params, optimizer, lr, loss, schedule = None):
    optimizer = optimizer.lower()
    if 'adam' in optimizer:
        optimizer = optim.Adam(lr = lr, params = params, weight_decay=1e-5)
    elif 'sgd' in optimizer:
        optimizer = optim.SGD(lr = lr, momentum=0.9, params = params)
    else:
        print('optimizer {} not yet supported'.format(optimizer))

    scheduler = None
    if schedule is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, math.exp(schedule))
    
    loss = loss.lower()
    if 'cont' in loss:
        criterion = ContrastiveLoss()
    elif 'trip' in loss:
        criterion = TripletLoss()
    else:
        print('loss {} not supported'.format(loss))

    return (optimizer, criterion, scheduler)

def train(model, dataset, lr, optimizer, epochs, loss, batch_size, schedule = None):
    optimizer, criterion, scheduler = parse_parameters(model.parameters(), optimizer, lr, loss, schedule)
    print('starting training process')
    if 'trip' in loss.lower():
        pass#model = train_triplet(model, optimizer, criterion, scheduler, epochs, dataset, batch_size)
    elif 'cont' in loss.lower():
        model = train_contrastive(model, optimizer, criterion, scheduler, epochs, dataset, batch_size)
    return model

def train_contrastive(model, optimizer, criterion, scheduler, epochs, dataset, batch_size):
    for epoch in range(epochs):
        dataset.mine_negatives(model)
        loss_mean = 0
        ix = 0
        for item in tqdm(dataset):
            if ix < batch_size:
                ix += 1
                anchor, image, label = item
                anchor = Variable(anchor).cuda()
                image = Variable(image).cuda()
                label = Variable(torch.Tensor([[label]])).cuda()
                o1 = model(anchor)
                o2 = model(image)
                loss = criterion(o1, o2, label)
                loss_mean += loss.data[0]
                loss.backward()
            else:
                optimizer.step()
                optimizer.zero_grad()
                ix = 0
        scheduler.step()
        print('epoch # {}, mean loss {}'.format(epoch, loss_mean/len(dataset)))
    return model



    
