
# We develop the DLGNN code based on the N3Net code from
# https://github.com/visinf/n3net/blob/master/src_denoising/models/non_local.py
#and IGNN code from https://github.com/sczhou/IGNN
# Thanks Tobias Plötz, Stefan Roth and Shangchen Zhou for their code.
#
from ptflops import get_model_complexity_info
#from models.submodules import *
#from models.VGG19 import VGG19
import sys
sys.path.append(".")

from config import cfg
#device=torch.device("cpu")
import torch
import torch.nn as nn
import torchvision.models
import warnings
# import submodules
import sys

sys.path.append(".")
import models.submodules
from models.submodules import *


# from models.submodules import *
# device=torch.device("cpu")

class VGG19(nn.Module):
    def __init__(self, feature_list=[2, 7, 14], requires_grad=True):
        super(VGG19, self).__init__()
        '''
        'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        ]
        use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = feature_list
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1] + 1])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)

        self.sub_mean = MeanShift(1.0, vgg_mean, vgg_std)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x : The input RGB tensor normalized to [0, 1].
        """
        x = x / 255.
        if torch.any(x < 0.) or torch.any(x > 1.):
            warnings.warn('input tensor is not normalize to [0, 1].')

        x = self.sub_mean(x)
        features = []

        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
            if i == self.feature_list[-1]:
                if len(self.feature_list) == 1:
                    return features[0]
                else:
                    return features


class DLGNN(nn.Module):
    def __init__(self):
        super(DLGNN, self).__init__()
        kernel_size = 3 #
        n_resblocks = cfg.NETWORK.N_RESBLOCK   #32
        n_feats = cfg.NETWORK.N_FEATURE   #256
        n_neighbors = cfg.NETWORK.N_REIGHBOR   #5
        scale = cfg.CONST.SCALE   #4
        if cfg.CONST.SCALE == 4:
            scale = 2  #
        window = cfg.NETWORK.WINDOW_SIZE  #30
        gcn_stride = 2  #
        patch_size = 3  #l=3

        self.sub_mean = MeanShift(rgb_range=cfg.DATA.RANGE, sign=-1)#DATA.RANGE:255  
        self.add_mean = MeanShift(rgb_range=cfg.DATA.RANGE, sign=1)#-----------------------------------------------

        self.vggnet = VGG19([3])

        self.graph = Graph(scale, k=n_neighbors, patchsize=patch_size, stride=gcn_stride, 
            window_size=window, in_channels=256, embedcnn=self.vggnet)

        # define head module
        self.head = conv(3, n_feats, kernel_size, act=False)
        # middle 16
        pre_blocks = int(n_resblocks//2) #graphagg模块之前的res模块数量
        # define body module
        m_body1 = [
            ResBlock(
                n_feats, kernel_size, res_scale=cfg.NETWORK.RES_SCALE 
            ) for _ in range(pre_blocks)#graphagg模块之前的resblock
        ]

        m_body2 = [
            ResBlock(
                n_feats, kernel_size, res_scale=cfg.NETWORK.RES_SCALE 
            ) for _ in range(n_resblocks-pre_blocks)#graphagg模块之后的resblock
        ]

        m_body2.append(conv(n_feats, n_feats, kernel_size, act=False))

        fuse_b = [
            conv(n_feats*2, n_feats, kernel_size),
            conv(n_feats, n_feats, kernel_size, act=False) # act=False important for relu!!!
        ]

        fuse_up = [
            conv(n_feats*2, n_feats, kernel_size),
            conv(n_feats, n_feats, kernel_size)        
        ]

        if cfg.CONST.SCALE == 4:
            #尾部
            #如果scale=4 需要再进行一个上采样
            m_tail = [
                upsampler(n_feats, kernel_size, scale, act=False),#scale=2
                conv(n_feats, 3, kernel_size, act=False)  # act=False important for relu!!!
            ]
        else:
            m_tail = [
                conv(n_feats, 3, kernel_size, act=False)  # act=False important for relu!!!
            ]            

        self.body1 = nn.Sequential(*m_body1)
        self.gcn = GCNBlock(n_feats, scale, k=n_neighbors, patchsize=patch_size, stride=gcn_stride)

        self.fuse_b = nn.Sequential(*fuse_b)

        self.body2 = nn.Sequential(*m_body2)
       
        self.upsample = upsampler(n_feats, kernel_size, scale, act=False)
        self.fuse_up = nn.Sequential(*fuse_up)

        self.tail = nn.Sequential(*m_tail)


    def forward(self, x_son, x):

        score_k, idx_k, diff_patch = self.graph(x_son, x) # ELS  El
        idx_k = idx_k.detach()
        if cfg.NETWORK.WITH_DIFF:#true
            diff_patch = diff_patch.detach()

        x = self.sub_mean(x)
        x0 = self.head(x)  #conv
        x1 = self.body1(x0)  #resblock  FL
        x1_lr, x1_hr = self.gcn(x1, idx_k, diff_patch)  #graphagg   得到hr 和lr
        x1 = self.fuse_b(torch.cat([x1, x1_lr], dim=1))   #将lr和den(x1_hr)进行concat  FL'
        x2 = self.body2(x1) + x0  #resblock 
        x = self.upsample(x2)  
        x = self.fuse_up(torch.cat([x, x1_hr], dim=1))   #x和hl进行concat 
        x= self.tail(x)   #conv
        x = self.add_mean(x)  #

        return x 


