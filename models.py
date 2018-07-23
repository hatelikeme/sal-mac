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

class Joint_Model(nn.Module):
    def __init__(self, sal_model, ret_model):
        super(Joint_Model, self).__init__()
        self.sal_model = sal_model
        self.ret_model = ret_model

    def forward(self, x):
        x = self.sal_model(x)
        x = self.ret_model(x)
        return x

class SalRet_Model(nn.Module):
    def __init__(self, model):
        super(SalRet_Model, self).__init__()
        self.model = model

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 7, stride = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size = 5, stride = 2)
        self.bn5 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size = 7, stride = 3)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        image = x
        x = F.elu(self.bn1(self.conv1(x)))
        x, inds1 = self.pool(x)
        s1 = x.size()

        x = F.elu(self.bn2(self.conv2(x)))
        x, inds2 = self.pool(x)
        s2 = x.size()

        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.deconv1(x)))
        x = F.upsample_bilinear(x, size = (s2[2], s2[3]))

        x = F.elu(self.bn5(self.deconv2(x)))
        x = F.upsample_bilinear(x, size = (s1[2], s1[3]))

        x = F.sigmoid(self.deconv3(x))
        ims = image.size()
        x = F.upsample_bilinear(x, size = (ims[2], ims[3]))
        sal_app = x * image
        return self.model(sal_app)
        
        
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
    elif 'mbd' in model_type:
        print('model {} is not yet supported'.format(model_type))

