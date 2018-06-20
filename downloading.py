import os
import tarfile
import urllib


PA_DATADIR = 'data/paris6k'
PA_GTDIR = 'data/paris_gt'
OX_DATADIR = 'data/oxford5k'
OX_GTDIR = 'data/oxford_gt'


def datasets_download(dataset, transform, model):
    if 'paris' in dataset:
        datapath = PA_DATADIR
        gtpath = PA_GTDIR
    elif 'oxford' in dataset:
        datapath = OX_DATADIR
        gtpath = OX_GTDIR
    else:
        print('validation dataset {} is not supported'.format(dataset))

    check_dataset(dataset, datapath)

def check_dataset(dataset, datapath):
    if not os.path.exists(datapath):
        print('downloading dataset')
        download_dataset(dataset)

def download_and_extract(url, path):
    file_tmp = urllib.request.urlretrieve(url, filename=None)[0]
    tar = tarfile.open(file_tmp)
    tar.extractall(path = path)

def download_dataset(dataset):
    if 'paris' in dataset:
        download_and_extract('http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz', 'data/paris6k')
        download_and_extract('http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz', 'data/paris6k')
        download_and_extract('http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz', 'data/paris_gt')
    elif 'oxford' in dataset:
        download_and_extract('http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz', 'data/oxford5k')
        download_and_extract('http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz', 'data/oxford_gt')