'测试.py'
import os,argparse
import numpy as np
from PIL import Image
#from models import *
import torch
from metrics import psnr,MSE
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from option1 import opt as opts
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image

#from BIFPN_Unet_1 import Unet_MDTA
#from net.bifpn_mdta import Unet_MDTA
#from net.Unet_mdta import Unet_MDTA
#from BIFPN_Unet_plus_2 import myself_train_Unet_MDTA as Unet_MDTA##(最好的结果)
#from no_feaformer import myself_train_Unet_MDTA as Unet_MDTA
from mdoel_new import model_NEW as UIE_PVtransformer
from torchvision.utils import make_grid
abs=r'E:\xzx\dataset\LSUI/'

def tensorShow(tensors,titles=['test_bad']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.ion()
        plt.pause(1)
        plt.close()
def crop_image(img):
    #width = img.size[0]  # 获取宽度
    #height = img.size[1]
    #width,height=int(width//16)*16, int(height//16)*16
    #return width,height
    return 256,256
parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
#parser.add_argument('--resize',type=int,default=240,help='resize dataset')
parser.add_argument('--blocks',type=int,default=5,help='residual_blocks')
#parser.add_argument('--test_imgs',type=str,default='/UIEB',help='Test imgs folder')
#parser.add_argument('--test_imgs',type=str,default='/LSUI_test/train_bad',help='Test imgs folder')
#parser.add_argument('--test_imgs',type=str,default='/EUVP_test_1000/train_bad',help='Test imgs folder')
parser.add_argument('--test_imgs',type=str,default='test_bad_crop_out',help='Test imgs folder')
parser.add_argument('--reference_imgs',type=str,default='reference_out',help='Reference imgs folder')
#parser.add_argument('--reference_imgs',type=str,default='EUVP_test_1000/label_best',help='Reference imgs folder')
#parser.add_argument('--reference_imgs',type=str,default='LSUI_test/label_best',help='Reference imgs folder')
#parser.add_argument('--reference_imgs',type=str,default='UIEB_reference_890',help='Reference imgs folder')
parser.add_argument('--imat',type=str,default='.jpg',help='image format')
opt=parser.parse_args()
dataset=opt.task
gps=2
blocks=5
resize=256
dim=80
#dim=48
#dim=64
img_reference_dir=abs+opt.reference_imgs+'/'
img_dir=abs+opt.test_imgs+'/'

#output_dir='C:/WBB/UNET_MDTA/test_data/test_get/myself_model_UAT/test_R90_get/'
#output_dir='C:/WBB/UNET_MDTA/test_data/test_get/myself_model_UAT/EUVP_get/'
#output_dir='C:/WBB/UNET_MDTA/test_data/test_get/myself_model_UAT/ufo_120_get/'
output_dir=r'E:\xzx\训练代码7.30\test_get\R90/'#
#output_dir='C:/WBB/UNET_MDTA/test_data/test_get/myself_model_UAT/UIEB_get/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
device='cuda' if torch.cuda.is_available() else 'cpu'
epochs = [200]
##epochs = [6, 7, 8, 9, 10]
for epoch in list(epochs):
    ssims=[]
    psnrs=[]
    mses = []
    output_dir1=output_dir+'test_out'+str(epoch)+'/'
    reference_crop_dir=output_dir+'reference_out'+'/'
    test_bad_crop_dir = output_dir + 'test_bad_crop_out'+ '/'
    test_out_mse_psnr_dir=output_dir+'test_get_psnr_ssim_MSE' + str(epoch) + '/'
    if not os.path.exists(output_dir1):
        os.mkdir(output_dir1)
    if not os.path.exists(reference_crop_dir):
        os.mkdir(reference_crop_dir)
    if not os.path.exists(test_bad_crop_dir):
        os.mkdir(test_bad_crop_dir)
    if not os.path.exists(test_out_mse_psnr_dir):
        os.mkdir(test_out_mse_psnr_dir)
    model_dir = f'./trained_models/UIE_PVtransformer_4/{opts.net}_epoch_{epoch}.pk'
    #model_dir = 'C:/WBB/UNET_MDTA/myself_model_UAT/net/' + f'trained_models/{dataset}_train_myself_model_{epoch}.pk'
    ckp = torch.load(model_dir, map_location=device)
    net = UIE_PVtransformer(dim=48)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()
    files=os.listdir(img_dir)
    print('files length:',len(files))
    for file in files:
        test_bad_old = Image.open(img_dir + file)
        width,height = crop_image(test_bad_old)
##        print('test_bad_old.shape:',test_bad_old.shape)
        test_bad =test_bad_old.resize((width,height),Image.BILINEAR)##resize输入图片
        #box=(0,0,width,height)
        #test_bad = test_bad_old.crop(box)
        test_bad.save(test_bad_crop_dir+file.split('.')[0] + opt.imat)
        reference_old= Image.open(img_reference_dir + file)
        reference = reference_old.resize((width,height),Image.BILINEAR)##resize输入图片
        reference.save(reference_crop_dir+file.split('.')[0] + opt.imat)

        reference1 = tfs.Compose([
            tfs.ToTensor()])(reference)[None, ::]
        test_bad1 = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])##Pytorch图像预处理时，通常使用transforms.Normalize(mean, std)对图像按通道进行标准化，即减去均值，再除以方差。
            # 这样做可以加快模型的收敛速度。其中参数mean和std分别表示图像每个通道的均值和方差序列。
        ])(test_bad)[None, ::]
        test_bad_no = tfs.ToTensor()(test_bad)[None, ::]
        with torch.no_grad():
            pred = net(test_bad1)
            #del pred1
            ##print('pred:',pred)
        pred = torch.tensor(pred)
        reference = torch.tensor(reference1)
        ts = torch.squeeze(pred.clamp(0, 1).cpu())##torch.squeeze()函数：删除不必要的维度，提出了shape返回结果中的1
        #print('ts.shape:',ts.shape)##clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        reference = torch.squeeze(reference.clamp(0, 1).cpu())


        #tensorShow([test_bad_no, pred.clamp(0, 1).cpu()], ['test_bad', 'pred'])
        #print('ts.shape:', ts.shape)
        vutils.save_image(ts, output_dir1 + file.split('.')[0] + opt.imat)   ##针对png图片
        reference = reference.unsqueeze(0)##reference和pred都在（0，1）之间
        #print('\t reference：', reference)
        ##print('reference.shape:', reference.shape)
        ts = ts.unsqueeze(0)
        # print('ts.shape:', ts.shape)
        with torch.no_grad():
            #ssim1 = ssim(reference, ts)
            ssim1 = ssim(reference, ts,data_range=1,size_average=False).to(device).item()  ##保存预测和目标之间的ssim值,
            # size_average (bool, optional): 如果size_average=True，所有图像的ssim将被平均为一个标量
            psnr1 = psnr(reference, ts)  ##保存预测和目标之间的psnr值
            mse1 = MSE(reference,ts)
        #ssims.append(ssim1)
        #psnrs.append(psnr1)
        aa = file.split('.')[0]
        tss = vutils.make_grid(#官网说的目的是：Make a grid of images.,组成图像的网络，其实就是将多张图片组合成一张图片
            [torch.squeeze(test_bad_no.cpu()), torch.squeeze(reference.cpu()), torch.squeeze(ts.clamp(0, 1).cpu())])
        vutils.save_image(tss,test_out_mse_psnr_dir+f'{aa}_{psnr1:.4}_{ssim1:.4}.png')
        del pred
        del ts
        del reference

        print(f'\nfile name :{file} |ssim:{ssim1:.4f}| psnr:{psnr1:.4f}| MSE:{mse1:.4f}')
        ssims.append(ssim1)
        psnrs.append(psnr1)
        mses.append(mse1)

    ssim_eval =np.mean(ssims)
    psnr_eval = np.mean(psnrs)
    #mse_sum = np.sum(mses)
    mse_eval = np.mean(mses)
    print('SSIM.len:',len(ssims))
    print(f'\nepoch :{epoch} |ssim_eval:{ssim_eval:.4f}| psnr_eval:{psnr_eval:.4f}| mse_eval:{mse_eval:.4f}')
    #print(f'\nepoch :{epoch} |mse_eval:{mse_eval:.4f}| mse_sum:{mse_sum:.4f}')
    #print('\nssims=',np.array(ssims))
    #print('\npsnrs=', np.array(psnrs))
    #print(f'\nepoch :{epoch} |ssim_eval:{ssim_eval:.4f}| psnr_eval:{psnr_eval:.4f}')\npsnrs= {np.array(psnrs):.4f}'
        # vutils.save_image(ts, output_dir + im.split('.')[0] + '.jpg')##针对jpg图片
