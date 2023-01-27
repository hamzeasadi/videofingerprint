import numpy as np
import torch
from torch import nn as nn
import model as m
import utils
import conf as cfg
import os
from torch import optim
from torch.utils.data import DataLoader
import engine
import datasetup as ds
import argparse
import cv2
from matplotlib import pyplot as plt



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def coords(H, W):
    xcoord = torch.ones(size=(H, W), dtype=torch.float32)
    ycoord = torch.ones(size=(H, W), dtype=torch.float32)
    for i in range(H):
        xcoord[i, :] = 2*(i/H) - 1
    for j in range(W):
        ycoord[:, j] = 2*(j/W) - 1 
    
    coord = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)

    return coord


def read_oneimage(imgpath):
    img = (cv2.imread(imgpath)/255)
    imgt = torch.from_numpy(img).permute(2, 0, 1)
    coord = coords(H=720, W=1280)
    imgt = torch.cat((imgt, coord), dim=0)
    return imgt.unsqueeze(dim=0), img


def main():
    kt = utils.KeepTrack(path=cfg.paths['model'])
    mn = 'videofingerprint1_0.pt'
    mn1 = 'videofingerprint1_1.pt'

    imgname1 = 'video7out1047.bmp'
    imgname2 = 'video7out1417.bmp'
    imgname3 = 'video6out154.bmp'
    iframepath1 = os.path.join(cfg.paths['testdata'], 'D16_Huawei_P9Lite', imgname3)
    iframepath2 = os.path.join(cfg.paths['testdata'], 'D16_Huawei_P9Lite', imgname1)

    imgt1, img1 = read_oneimage(iframepath1)
    imgt2, img2 = read_oneimage(iframepath2)
    
    print(imgt1.shape, imgt2.shape)
    print(imgt1)
    print(imgt2)
   
    Net = m.VideoPrint(inch=3, depth=25)
    state = kt.load_ckp(fname=mn)
    Net.to(dev)
    Net.load_state_dict(state['model'])
    Net.eval()
    print(state['minerror'])
    out1, out2 = Net(imgt1, imgt2)

    out1 = out1.squeeze().numpy()
    out2 = out2.squeeze().numpy()
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 16))


    out = [out1, out2]
    img = [img1, img2]

    for i in range(2):
        axs[i, 0].imshow(img[i])
        axs[i, 0].axis('off')
        axs[i, 1].imshow(out[i])
        axs[i, 1].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()




    
    



if __name__ == '__main__':
    main()