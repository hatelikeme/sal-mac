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

def validate(model, dataset):
    

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