import numpy as np
import torch
from torch import nn as nn
import model as m
import utils
import conf as cfg
import os, random
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



def createsample(folderpath):
    listimgs = os.listdir(folderpath)
    listimgs = cfg.ds_rm(listimgs)
    subimages = random.sample(listimgs, 4)
    imgs = []
    for i, imgname in enumerate(subimages):
        imgpath = os.path.join(folderpath, imgname)
        img = cv2.imread(imgpath)
        if i%2==0:
            img[100:200, 200:400, :] = img[-100:-200, -200:-400, :]
        imgt = torch.from_numpy(img).permute(2, 0, 1)
        coord = coords(H=720, W=1280)
        imgt = torch.cat((imgt, coord), dim=0) 
        imgs.append((img, imgt))
    
    return imgs


def main():
    kt = utils.KeepTrack(path=cfg.paths['model'])
    mn = 'videofingerprint1_0.pt'
    mn1 = 'videofingerprint1_2.pt'

    folderpath = os.path.join(cfg.paths['testdata'], 'D16_Huawei_P9Lite')
    imgpairs = createsample(folderpath=folderpath)

    Net = m.VideoPrint(inch=3, depth=25)
    state = kt.load_ckp(fname=mn1)
    Net.to(dev)
    Net.load_state_dict(state['model'])
    Net.eval()

    Noises = []
    for i in range(0, 4, 2):
        img1, img2 = imgpairs[i][1], imgpairs[i+1][1]
        out1, out2 = Net(img1, img2)
        Noises.append(out1.squeeze().detach().numpy())
        Noises.append(out2.squeeze().detach().numpy())

    
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 9))

    for i in range(4):
        axs[i, 0].imshow(imgpairs[i][0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(Noises[i], cmap='gray')
        axs[i, 1].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('output.png')
    plt.show()




    
    



if __name__ == '__main__':
    main()