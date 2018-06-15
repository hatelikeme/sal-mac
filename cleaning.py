from PIL import Image
import os
import pandas as pd

def clean(filedir, labels):
    df = pd.read_csv(labels, sep = ';', dtype = str)
    df.columns = ['image', 'label']
    inds = df['image'].apply(lambda x: check_image(x, filedir))
    df = df[inds]
    groups = df.groupby('label')
    df = groups.filter(lambda x: len(x) > 1)
    return df

def check_image(im, imdir):
    impath = os.path.join(imdir, str(im))
    if os.path.exists(impath):
        try:
            im = Image.open(impath)
            w, h = im.size
            if w > 100 and h > 100:
                return True
            return False
        except:
            return False
    else:
        return False
