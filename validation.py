import os
import requests
import shutil
import tarfile
import urllib
import subprocess

import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

PA_DATADIR = 'data/paris6k'
PA_GTDIR = 'data/paris_gt'
OX_DATADIR = 'data/oxford5k'
OX_GTDIR = 'data/oxford_gt'
    

class ValidationDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images = os.listdir(root)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def transform_fn(self, path):
        im = Image.open(os.path.join(self.root, path))
        return self.transform(im).unsqueeze(0)
    
    def __getitem__(self, idx):
        rv = self.images[idx]
        return (self.transform_fn(rv), rv)


def init_validation(dataset, transform, model):
    if 'paris' in dataset:
        datapath = PA_DATADIR
        gtpath = PA_GTDIR
    elif 'oxford' in dataset:
        datapath = OX_DATADIR
        gtpath = OX_GTDIR
    else:
        print('validation dataset {} is not supported'.format(dataset))

    check_dataset(dataset, datapath)
    data = ValidationDataset(datapath, transform)
    mAP = calculate_map(data, transform, model, gtpath, datapath)
    return mAP

def check_dataset(dataset, datapath):
    if not os.path.exists(os.path.join('data', datapath)):
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

def calculate_map(test_data, vgg_transform, model, gt_dir, datadir):
    db_names, db_codes = compute_db(test_data, model)
    aps = []
    for file in os.listdir(gt_dir):
        if 'query' in file:
            q_name, rect = process_query(os.path.join(gt_dir, file))
            vgg_q = single_im_loader(os.path.join(datadir, q_name), rect, vgg_transform)
            code = model(Variable(vgg_q, requires_grad = False).cuda())
            distlist = generate_r_list(code.cpu().data.numpy(), db_codes)
            gen_txt_file(distlist, db_names)
            query_name = file.replace('_query.txt', '')
            ap = compute_ap(query_name)
            aps.append(ap)
    return np.mean(aps)

def compute_ap(query):
    process = subprocess.Popen('./compute_ap {} ranked_list.txt'.format(query), shell = True, stdout=subprocess.PIPE,
                              cwd = './oxford_gt')
    output = process.stdout.read()
    return float(output)

def gen_txt_file(distlist, names):
    act = []
    for idx in distlist:
        act.append(names[idx])
    with open('./oxford_gt/ranked_list.txt', 'w') as f:
        for item in act:
            f.write(item.replace('.jpg', '') + '\n')

def generate_r_list(query, codes):
    distances = []
    for code in codes:
        dist = cosine(code.squeeze(), query.squeeze())
        distances.append(dist)
    sorted_indicies = np.argsort(distances)
    return sorted_indicies

def single_im_loader(impath, rect, vgg_transform):
    im = Image.open(impath)
    im = np.asarray(im)
    im = im[rect[0]: rect[2], rect[1]: rect[3]]
    im = Image.fromarray(im)
    vgg_im = vgg_transform(im)
    return vgg_im.unsqueeze(0).cuda()

def process_query(filename):
    with open(filename) as f:
        line = f.readlines()[0]
        q_split = line.split(' ')
        q_name = q_split[0].replace('oxc1_', '') + '.jpg'
        x1, y1, x2, y2 = int(float(q_split[1])), int(float(q_split[2])), int(float(q_split[3])), int(float(q_split[4]))
        return q_name, (x1, y1, x2, y2)

def compute_db(test_data, model):
    names = []
    codes = []
    idx = 0
    for vgg_im, filename in tqdm(test_data):
        vgg_im = Variable(vgg_im).cuda()
        output = model(vgg_im)
        names.append(filename)
        code = output.cpu().data.numpy()
        codes.append(code)
        idx += 1
    return names, codes