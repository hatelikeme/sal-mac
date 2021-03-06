import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from models import get_unsupervised_model, load_model, Retrieval_Model, SalRet_Model
from datasets import Training_Contrastive_Dataset
from losses import ContrastiveLoss
from training import train
from validation import calculate_map
from downloading import datasets_download
from cleaning import clean

from PIL import ImageFile

import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('SAL-MAC training and testing')

parser.add_argument('--data', metavar='DIR', default='../landmarks-clean/data_train')
parser.add_argument('--labels', default = '../landmarks-clean/labels.csv')
parser.add_argument('--arch', '-a', metavar='ARCH',default='alexnet')
parser.add_argument('--pretrained', '-pt', metavar='PT', default=False)
parser.add_argument('--train', '-t', metavar='TRAIN', default=True)
parser.add_argument('--pool', default = 'mac')
parser.add_argument('--loss', default='contrastive')
parser.add_argument('--epochs', default=120)
parser.add_argument('--lr', '-lr', default=0.0001)
parser.add_argument('--batch_size', '-bs', default=32)
parser.add_argument('--optimizer', '-o', default = 'adam')
parser.add_argument('--saliency', '-s', default = '', type = str)
parser.add_argument('--salsupervised', '-ss', default = '')
parser.add_argument('--val_dataset', default = 'oxford5k')

transform = transforms.Compose([
    transforms.ToTensor()
])

def main():
    args = parser.parse_args()

    arch = args.arch
    pretrained = args.pretrained
    pool = args.pool
    model = Retrieval_Model(load_model(arch, pretrained), pool)
    if args.salsupervised is not '':
        model = SalRet_Model(model).cuda()
    sal_model = None
    #if args.saliency is not '' and args.:
    #    sal_model = get_unsupervised_model(args.saliency)
    if args.train:
        datadir = args.data
        loss = args.loss
        epochs = args.epochs
        dataset = args.labels
        datadir = args.data
        lr = args.lr
        batch_size = args.batch_size
        dataset = Training_Contrastive_Dataset(dataset, datadir, transform, args.saliency)
        optimizer = args.optimizer
        train(model, dataset, lr, optimizer, epochs, loss, batch_size, 0.001)
    datasets_download(args.val_dataset)
    mAP = calculate_map(model, transform, 'data/oxford_gt', 'data/oxford5k', sal_model)
    print('mAP: {}'.format(mAP))

if __name__ == '__main__':
    main()