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
    return imgt


def main():
    kt = utils.KeepTrack(path=cfg.paths['model'])
    imgname1 = 'video7out1047.bmp'
    imgname2 = 'video7out1417.bmp'
    imgname3 = 'video6out154.bmp'
    iframepath1 = os.path.join(cfg.paths['testdata'], 'D16_Huawei_P9Lite', imgname3)
    iframepath2 = os.path.join(cfg.paths['testdata'], 'D16_Huawei_P9Lite', imgname1)

    imgt1 = read_oneimage(iframepath1)
    imgt2 = read_oneimage(iframepath2)
    
    print(imgt1.shape, imgt2.shape)
    print(imgt1)
    print(imgt2)
    # mn = ''
    # Net = m.VideoPrint(inch=3, depth=25)
    # state = kt.load_ckp(fname=mn)
    # Net.to(dev)
    # Net.load_state_dict(state['model'])
    # print(state['minerror'])
    # iframe = cv2.imread()




    
    



if __name__ == '__main__':
    main()