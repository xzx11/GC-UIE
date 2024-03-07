from GCN1 import NEW,pvig_ti_224_gelu,threeC
from GCN1 import _cfg
import torch
import time
import torch.autograd.profiler as profiler
import cv2
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import numpy as np
from PIL import Image

default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}



def model_NEW(pretrained=False, **kwargs):
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
    model = NEW(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    #print('model=',model)
    return model


def get_flops(model, input_data):
    # 获取模型的总参数量
    total_params = sum([p.numel() for p in model.parameters()])

    # 将模型设置为评估模式
    model.eval()

    # 将输入数据传递到模型中，并记录前向传递的时间
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    elapsed_time = time.time() - start_time

    # 计算FLOPS
    flops = 2 * output.numel() * total_params / elapsed_time / 1e9

    return flops


if __name__== '__main__':
    start_time = time.time()
    x1 = torch.randn([1, 3, 256, 256])#1, 3, 256, 256
   # C = x1.shape[1]
    #attn = MBG_Transformer_upstage(depth=3,in_channels=C,out_channels=C//2)
    #attn=MBGVIT()
    #attn=Global_Local_Transformer_block(dim=x1.shape[1],num_heads=4,window_size=7)
    attn = model_NEW()
    total_params = sum(p.numel() for p in attn.parameters())  # 计算总的参数量
    print(f"Total number of parameters: {total_params}")  # 打印结果
    flops = get_flops(attn, x1)
    print(f"Model FLOPS: {flops:.2f} GFLOPS")

    x2 = attn(x1)
    print('x2=', x2.shape)
    print('time=',time.time()-start_time)

