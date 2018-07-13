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

class ValidationDataset(Dataset):
    def __init__(self, root, transform, sal_model):
        self.root = root
        self.images = os.listdir(root)
        self.transform = transform
        self.sal = sal_model
    
    def __len__(self):
        return len(self.images)
    
    def transform_fn(self, path):
        im = Image.open(os.path.join(self.root, path))
        if self.sal is not None:
            salmap = self.sal(os.path.join(self.root, path))
            salmap = salmap[...,np.newaxis]
            im = salmap * im
            im = Image.fromarray(im.astype('uint8'), 'RGB')
        return self.transform(im).unsqueeze(0)
    
    def __getitem__(self, idx):
        rv = self.images[idx]
        return (self.transform_fn(rv), rv)

def calculate_map(model, transform, gt_dir, datadir, sal_model = None):
    test_data = ValidationDataset(datadir, transform, sal_model)
    db_names, db_codes = compute_db(test_data, model)
    aps = []
    for file in os.listdir(gt_dir):
        if 'query' in file:
            q_name, rect = process_query(os.path.join(gt_dir, file))
            vgg_q = single_im_loader(os.path.join(datadir, q_name), rect, transform, sal_model)
            code = model(Variable(vgg_q, requires_grad = False).cuda())
            distlist = generate_r_list(code.cpu().data.numpy(), db_codes)
            gen_txt_file(distlist, db_names, gt_dir)
            query_name = file.replace('_query.txt', '')
            ap = compute_ap(query_name, gt_dir)
            aps.append(ap)
    return np.mean(aps)

def compute_ap(query, gt_dir):
    process = subprocess.Popen('./compute_ap {} ranked_list.txt'.format(query), shell = True, stdout=subprocess.PIPE,
                              cwd = gt_dir)
    output = process.stdout.read()
    return float(output)

def gen_txt_file(distlist, names, gt_dir):
    act = []
    for idx in distlist:
        act.append(names[idx])
    with open(os.path.join(gt_dir, 'ranked_list.txt'), 'wb') as f:
        for item in act:
            it = item.replace('.jpg', '') + '\n'
            f.write(it.encode())

def generate_r_list(query, codes):
    distances = []
    for code in codes:
        dist = cosine(code.squeeze(), query.squeeze())
        distances.append(dist)
    sorted_indicies = np.argsort(distances)
    return sorted_indicies

def single_im_loader(impath, rect, vgg_transform, sal_model = None):
    im = Image.open(impath)
    im = np.asarray(im)
    im = im[rect[0]: rect[2], rect[1]: rect[3]]
    if sal_model is not None:
        salmap = sal_model(im)
        salmap = salmap[..., np.newaxis]
        im = im * salmap
    im = Image.fromarray(im.astype('uint8'), 'RGB')
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
