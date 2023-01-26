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


def train(net: nn.Module, opt: optim.Optimizer, train_data: DataLoader, val_data: DataLoader, criterion: nn.Module, modelname: str, epochs):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        trainloss = engine.train_step(model=net, data=train_data, criterion=criterion, optimizer=opt)
        valloss = engine.val_step(model=net, data=val_data, criterion=criterion)
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")
        fname=f'{modelname}_{epoch}.pt'
        kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=1, fname=fname)







def main():
    Net = m.VideoPrint(inch=3, depth=25)
    crt = utils.OneClassLoss(batch_size=100, pairs=2, reg=0.1)
    opt = optim.Adam(params=Net.parameters(), lr=3e-3)
    
    



if __name__ == '__main__':
    main()