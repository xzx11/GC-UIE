# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import cv2
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from gcn_lib import Grapher, act_layer
import time
from einops import rearrange
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d

import patchify as pat

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def three_c(img_rgb):

    img_lab = rgb2lab(img_rgb)
    img_gray = rgb2gray(img_rgb)
    img_gray[img_gray > 0.847] = 0
    img_gray[img_gray <= 0.847] = 1

    gray_gauss = cv2.GaussianBlur(img_gray, (0, 0), 5)
    ch1, ch2, ch3 = cv2.split(img_lab)
    ch2_gauss = cv2.GaussianBlur(ch2, (0, 0), 100)
    img_lab[:, :, 1] = np.subtract(ch2, np.multiply(gray_gauss, ch2_gauss))

    ch3_gauss = cv2.GaussianBlur(ch3, (0, 0), 100)
    img_lab[:, :, 2] = np.subtract(ch3, np.multiply(gray_gauss, ch3_gauss))

    new = lab2rgb(img_lab)

    return new

class threeC(nn.Module):
    def __init__(self, in_dim=3, out_dim=3):
        super().__init__()
        self.para = nn.Parameter(torch.ones(256, 256))
        self.para1 = nn.Parameter(torch.ones(256, 256))
    def forward(self, x):
        one = torch.ones([1, 1, 256, 256]).cuda()
        x_r = x[:, 0, :, :].unsqueeze(1)
        x_g = x[:, 1, :, :].unsqueeze(1)
        x_b = x[:, 2, :, :].unsqueeze(1)
        x_r_average = torch.mean(x_r)
        x_g_average = torch.mean(x_g)
        x_b_average = torch.mean(x_b)

        I_rg_1 = (x_g_average - x_r_average) * (one - x_r) * x_g
        I_rb_1 = (x_b_average - x_r_average) * (one - x_r) * x_b

        I_rg = x_r + torch.mul(self.para, I_rg_1)
        I_rb = x_r + torch.mul(self.para1, I_rb_1)
        I = torch.cat((x_r,I_rg, I_rb), 1)
        return I


class FFN(nn.Module):#输入[1,3,256,256],输出[1,3,256,256],in_features=out_features=3
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        #print('x=', x.shape)
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x



class conv_cha(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x





class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 256 // 4, 256 // 4))
        HW = 256 // 4 * 256 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(conv_cha(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.reducecha = Seq(
            nn.Conv2d(96, 48, 1, bias=True),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 24, 1, bias=True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 3, 1, bias=True),
            nn.BatchNorm2d(3))
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        # self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
        # nn.BatchNorm2d(1024),
        # act_layer(act),
        # nn.Dropout(opt.dropout),
        # nn.Conv2d(1024, 1024, 1, bias=True)
        # nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        #x = self.reducecha(x)
        x = self.upsample(x)
        #x = F.adaptive_avg_pool2d(x, 1)
        #print('x=', x.shape)
        return x
        # self.prediction(x).squeeze(-1).squeeze(-1)


class FeedForward(nn.Module):
    def __init__(self, dim, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*3)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x

class SELayer(nn.Module):
    def __init__(self,channel ,reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        '''
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        '''
        self.layernorm = nn.LayerNorm(96)
        self.cst =  ChannelAttention(channel,1,False)
        self.ffn = FeedForward(dim=96)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)#.view(b, c)
        #y = self.fc(y).view(b, c, 1, 1)
        y = self.cst(y)
        '''
        y_1 = y.permute(0, 2, 3, 1).contiguous()
        y_1 = self.layernorm(y_1)
        y_1 = y_1.permute(0, 3, 1, 2).contiguous()
        #y_half = y + y_1
        y_21 = self.ffn(y_1)
        y_2 = y_21.permute(0, 2, 3, 1).contiguous()
        y_2 = self.layernorm(y_2)
        y_2 = y_2.permute(0, 3, 1, 2).contiguous()
        y = y_1 + y_2
        '''
        return x * y.expand_as(x)



class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__() # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False) # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size() # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)





class CNN3(nn.Module):  #起始head
    def __init__(self, in_dim =3, out_dim =3 , act='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, 5, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, b):
        x = self.conv1(b)
        x = self.conv3(x)
        x = self.conv5(x)
        x = x + b
        x = self.batchnorm(x)
        self.relu = nn.ReLU()
        return x




class CNN2(nn.Module):
    def __init__(self, in_dim =3, out_dim=3 , act='relu'):
        super().__init__()
        self.conv =  nn.Conv2d(in_dim, in_dim, 1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_dim, in_dim, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, in_dim, 5, stride=1, padding=2)
        self.convs = nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, b):
        x1 = self.relu(self.conv3(b))
        x2 = self.relu(self.conv5(b))
        x_half = x1 + x2
        x_half_1 = self.relu(self.conv5(x_half+x1)) #+ x1
        x_half_2 = self.relu(self.conv3(x_half+x2)) #+ x2
        x = x_half_1 + x_half_2 + b
        x = self.relu(self.convs(x))
        return x



'''
class CNN2(nn.Module):
    def __init__(self, in_dim =3, out_dim=3 , act='relu'):
        super().__init__()
        self.conv =  nn.Conv2d(in_dim, in_dim, 1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_dim, in_dim, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, in_dim, 5, stride=1, padding=2)
        self.convs = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, b):
        x1 = self.relu(self.conv5(b))
        x2 = self.relu(self.conv3(b)) + x1
        x3 = self.relu(self.conv5(x1))
        x4 = self.relu(self.conv3(x2)) + x3
        x5 = self.relu(self.conv5(x3))
        x6 = self.relu(self.conv3(x4)) + x5
        x = x2 + x4 + x6
        x =self.relu(self.convs(x))
        return x
'''


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x






class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1,bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))/np.sqrt(int(c/self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class HEAD(nn.Module): #256
    def __init__(self, in_dim=3, out_dim=48, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 48, 3, stride=1, padding=1),
            act_layer(act),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x1 = self.convs(x)
        x2 = x + x1
        x3 = self.convs(x2)
        x4 = x + x3
        return x4






class NEW(torch.nn.Sequential):
    def __init__(self,opt,**kwargs):
        super(NEW, self).__init__()
        class OptInit:
            def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
                self.k = 9  # neighbor num (default:9)
                self.conv = 'mr'  # graph conv layer {edge, mr}
                self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
                self.norm = 'batch'  # batch or instance normalization {batch, instance}
                self.bias = True  # bias of conv layer True or False
                self.dropout = 0.0  # dropout rate
                self.use_dilation = True  # use dilated knn or not
                self.epsilon = 0.2  # stochastic epsilon for gcn
                self.use_stochastic = False  # stochastic for gcn, True or False
                self.drop_path = drop_path_rate
                self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
                self.channels = [24, 48, 48, 96]  # number of channels of deep features
                #self.channels2 = [24, 48, 48, 96]
                self.n_classes = num_classes  # Dimension of out_channels
                self.emb_dims = 1024  # Dimension of embeddings
        opt = OptInit(**kwargs)
        self.head = CNN3()

        self.gcn = DeepGCN(opt)
        self.cnn1 = CNN2(in_dim=3,out_dim=96)
        self.cnn = CNN2(in_dim=96,out_dim=96)
        #self.gcn6 = DeepGCN(opt)
        #self.cnn6 = CNN3(in_dim = 3,out_dim=3)
        #self.gcn12 = DeepGCN(opt,in_dim = 3,out_dim=3)
        #self.cnn12 = CNN3(in_dim = 3,out_dim=3)
        #self.gcn24 = DeepGCN(opt,in_dim = 3,out_dim=3)
        #self.cnn24 = CNN3(in_dim = 3,out_dim=3)

        #self.convonemone = convonemone6(in_dim=6)
        self.downsample = Downsample(in_dim=3,out_dim=3)
        #self.gcn128 = DeepGCN128(opt)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upconv = upConv(in_channels=3,out_channels=3)
        self.eca_layer = eca_layer(channel=3)
        self.selayer = SELayer(channel=96)
        self.threec = threeC(in_dim=3,out_dim=3)
        self.reducecha = Seq(
            nn.Conv2d(96, 48, 1, bias=True),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 24, 1, bias=True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 3, 1, bias=True),
            nn.BatchNorm2d(3))
    def forward(self,inputs):
        #r = self.threec(inputs)
        #good
        r0 = inputs
        r0 = self.head(r0)
        #print('r0=',r0.shape)
        #r0=self.selayer(r0)
        r10 = self.gcn(r0)
        r11 = self.cnn1(r0)
        r11 = self.cnn(r11)
        r11 = self.cnn(r11)
        r1 =  r10 + r11
        #print('r1=',r1.shape)
        r = self.selayer(r1)
        r = self.reducecha(r)
        r = r + r0

        #tt
        '''
        rd = self.downsample(inputs)
        rg = self.gcn128(rd)
        rg = self.upsample(rg)
        r = r1 + rg
        
        rx1 = self.cnn(inputs)


        rout = torch.cat((r1, rx1), 1)
        rout =  self.convonemone(rout)


        
        r30 = self.gcn(r2)
        r31 = self.cnn(r2)
        r3 = r30 +r31
        #r3 = torch.cat((r3, r2), 1)  # [1,12,128,128]
        #r3 = self.convonemone(r3)  # [1,6,128,128]
        r40 = self.gcn(r3)
        r41 = self.cnn(r3)
        r4 = r40 + r41
        r4 = torch.cat((r4, r1), 1)  # [1,12,256,256]
        r4 = self.convonemone(r4)  # [1,6,256,256]
        r50 = self.gcn(r4)
        r51 = self.cnn(r4)
        r5 = r50 + r51
        r5 = torch.cat((r3, r5), 1)  # [1,12,256,256]
        r5 = self.convonemone(r5)  # [1,6,256,256]

        r60 = self.gcn(r5)
        r61 = self.cnn(r5)
        r6 = r61 + r60
        r6 = torch.cat((r2, r6), 1)  # [1,12,256,256]
        r6 = self.convonemone(r6)  # [1,6,256,256]

        r70 = self.gcn(r6)
        r71 = self.cnn(r6)
        r7 = r71 + r70
        r7 = torch.cat((r1, r7), 1)  # [1,12,256,256]
        r7 = self.convonemone(r7)  # [1,6,256,256]

        r80 = self.gcn(r7)
        r81 = self.cnn(r7)
        r8 = r80 + r81
        r8 = self.head(r8)
        
        k1 = self.pconv(r0)
        k2 = self.pconv(k1)
        k3 = self.pconv(k2)
        '''
        return r

@register_model
def pvig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
            self.channels = [24, 48, 48, 96]  # number of channels of deep features
            #self.channels2 = [24, 48, 48, 96]
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


def model_NEW( **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
            self.channels = [24, 48, 48, 96]  # number of channels of deep features
            #self.channels2 = [24, 48, 48, 96]
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            #self.in_dim = 3
    opt = OptInit(**kwargs)
    print(opt)
    model = NEW(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


if __name__== '__main__':
    start_time = time.time()
    x1 = torch.randn([1, 3 , 256 , 256])#1, 3, 256, 256
    C = x1.shape[1]
    #attn = MBG_Transformer_upstage(depth=3,in_channels=C,out_channels=C//2)
    #attn=MBGVIT()
    #attn=Global_Local_Transformer_block(dim=x1.shape[1],num_heads=4,window_size=7
    attn = Stem()
    #att1=model_NEW()
    #print(att1)
    x2 = attn(x1)
    print('x2=', x2.shape)
    print('time=',time.time()-start_time)
