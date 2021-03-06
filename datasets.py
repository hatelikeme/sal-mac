import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from skimage import io, transform
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from cleaning import clean
import torch

from models import get_saliency_rbd, binarise_saliency_map

class Training_Contrastive_Dataset():
    def __init__(self, labels, root, transform, salient = False, n_negatives = 5):
        self.root = root
        self.transforms = transform
        self.df = clean(root, labels)
        self.data = None
        self.n_negatives = n_negatives
        self.salient = salient

    def read_img(self, image):
        impath = os.path.join(self.root, image)
        image = Image.open(impath).convert('RGB')
        image = self.transforms(image)
        image = image.unsqueeze(0)
        return image

    def compute_code(self, image, model):
        image = Variable(self.read_img(image)).cuda()
        code = model(image)
        return code.cpu().data[0].numpy()

    def mine_negatives(self, model):
        self.data = []
        anchors = self.df.sample(n = len(self.df)//100)
        pot_neg = self.df.sample(n = len(self.df)//100)
        anchors['code'] = anchors['image'].apply(lambda x: self.compute_code(x, model))
        pot_neg['code'] = pot_neg['image'].apply(lambda x: self.compute_code(x, model))
        for item in anchors.itertuples():
            label = getattr(item, 'label')
            code = getattr(item, 'code')
            image = getattr(item, 'image')

            df_s = self.df[self.df['label'] == label]
            pot_pos = [x for x in df_s['image'].tolist() if x != image]
            pos = np.random.choice(pot_pos, 1, False)[0]

            pos_tp = (image, pos, 1)
            self.data.append(pos_tp)
            pot_neg_subset = pot_neg[pot_neg['label'] != label]
            mt = np.array(pot_neg_subset['code'].tolist()).transpose()

            scores = np.dot(code, mt)
            inds = np.argsort(scores)[::-1][:self.n_negatives]
            rows = pot_neg_subset.iloc[inds]
            neg = []
            for row in rows.itertuples():
                neg_tp = (image, getattr(row, 'image'), 0)
                neg.append(neg_tp)
            self.data.extend(neg)

    def __len__(self):
        return len(self.data)

    def apply_salmap(self, img, item):
        sal = get_saliency_rbd(os.path.join(self.root, item))
        sal = binarise_saliency_map(sal)
        idd = (sal == 0)
        idd = torch.from_numpy(idd.astype(np.uint8)).type(torch.FloatTensor)
        idd = idd.unsqueeze(0).unsqueeze(0)
        img = torch.mul(img, idd)
        return img

    def __getitem__(self, idx):
        item = self.data[idx]

        im1 = self.read_img(item[0])
        im2 = self.read_img(item[1])

        if self.salient:
            im1 = self.apply_salmap(im1, item[0])
            im2 = self.apply_salmap(im2, item[1])
        return (im1, im2, item[2])