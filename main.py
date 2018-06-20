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

from models import get_unsupervised_model, load_model, Retrieval_Model
from datasets import Training_Contrastive_Dataset
from losses import ContrastiveLoss
from training import train
from validation import init_validation

from PIL import ImageFile

import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('SAL-MAC training and testing')

parser.add_argument('--data', metavar='DIR', default='../landmarks-clean/data_train')
parser.add_argument('--labels', default = '../landmarks-clean/labels.csv')
parser.add_argument('--arch', '-a', metavar='ARCH',default='alexnet')
parser.add_argument('--pretrained', '-pt', metavar='PT', default=True)
parser.add_argument('--train', '-t', metavar='TRAIN', default=False)
parser.add_argument('--pool', default = 'mac')
parser.add_argument('--loss', default='contrastive')
parser.add_argument('--epochs', default=30)
parser.add_argument('--lr', '-lr', default=0.0001)
parser.add_argument('--batch_size', '-bs', default=32)
parser.add_argument('--optimizer', '-o', default = 'adam')
parser.add_argument('--saliency', '-s', default = True)
parser.add_argument('--sal-supervised', '-ss', default = False)
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

    sal_model = None
    if args.saliency is not None:
        sal_model = get_unsupervised_model('rbd')
    print(args.train)
    if args.train:
        print('we here somehow')
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
    mAP = init_validation(args.val_dataset, transform, model)
    print('mAP: {}'.format(mAP))
    

if __name__ == '__main__':
    main()