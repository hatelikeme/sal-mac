import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from rbd import get_saliency_rbd, get_saliency_ft
from rbd import binarise_saliency_map

class Retrieval_Model(nn.Module):
    def __init__(self, model, pool):
        super(Retrieval_Model, self).__init__()
        self.pool = self.process_pooling(pool)
        self.model = model

    def process_pooling(self, pool):
        pool = pool.lower()
        if 'mac' in pool or 'max' in pool:
            pool_mod = lambda x: F.normalize(x.max(2)[0].max(2)[0], dim = 1, p = 2)
        elif 'spoc' in pool or 'sum' in pool:
            pool_mod = lambda x: F.normalize(x.sum(2).sum(2), dim = 1, p = 2)
        return pool_mod

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        return x

def load_model(arch, pretrained):
    arch = arch.lower()
    if 'vgg' in arch:
        model = models.vgg16(pretrained = pretrained)
        model = nn.Sequential(*list(model.features.children())[:-1])
    elif 'alexnet' in arch:
        model = models.alexnet(pretrained = pretrained)
        model = nn.Sequential(*list(model.features.children())[0:12])
    else:
        print('Other architectures not supported')
    return model.cuda()

def get_supervised_model():
    pass
    
def get_unsupervised_model(model_type):
    model_type = model_type.lower()
    if 'rbd' in model_type:
        return get_saliency_rbd
    elif 'ft' in model_type:
        return get_saliency_ft

