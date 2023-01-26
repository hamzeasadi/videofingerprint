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







def main():
    pass



if __name__ == '__main__':
    main()