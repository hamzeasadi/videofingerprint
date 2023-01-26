import conf as cfg
import os, random
import torch
from torch import nn
import cv2
import itertools as it
from torch.utils.data import Dataset, DataLoader




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def datasettemp(iframefolders):
    folders = os.listdir(iframefolders)
    folders = cfg.ds_rm(folders)
    temp = dict()
    i = 0
    for folder in folders:
        for h in range(8, 720-64, 64):
            for w in range(0, 1280, 64):
                patchid = f'patch_{i}'
                temp[patchid] = (folder, h, w)
                i+=1
    return temp



def get2patch(folderpath, H=720, W=1280):

    pass



class VideoNoiseSet(Dataset):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.datapath = datapath
        self.temp = datasettemp(iframefolders=datapath)
        self.patches = list(self.temp.keys())

    def crop(self, img, h, w):
        newimg = img[h:h+64, w:w+64, :]
        return newimg

    def creatcords(self, h, w, H, W):
        """
        the normalization depends on the input normalization
        I HAVE TO APPLY IT LATER
        """
        xcoord = torch.ones(size=(64, 64), dtype=torch.float32, device=dev)
        ycoord = torch.ones(size=(64, 64), dtype=torch.float32, device=dev)
        for i in range(h, h+64):
            xcoord[i-h, :] = 2*(i/H) -1

        for j in range(w, w+64):
            ycoord[:, j-w] = (2*j/W) - 1

        coords = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)
        return coords



    def get4path(self, patchid, H=720, W=1280):
        folder, h, w = patchid
        imgspath = os.path.join(self.datapath, folder)
        imgs = os.listdir(imgspath)
        imgs = cfg.ds_rm(imgs)
        subimgs = random.sample(imgs, 12)
        img12 = [cv2.imread(os.path.join(imgspath, i)) for i in subimgs]
        img12crop = [self.crop(img=im, h=h, w=w) for im in img12]
        for j in range(0, 12, 3):
            img12crop[j][:, :, 1] = img12crop[j+1][:, :, 0]
            img12crop[j][:, :, 2] = img12crop[j+2][:, :, 0]
        
        pairone = torch.cat((torch.from_numpy(img12crop[0]).unsqueeze(dim=0), torch.from_numpy(img12crop[3]).unsqueeze(dim=0)), dim=0)
        pairtwo = torch.cat((torch.from_numpy(img12crop[6]).unsqueeze(dim=0), torch.from_numpy(img12crop[9]).unsqueeze(dim=0)), dim=0)
        
        return pairone, pairtwo

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        subpatches = random.sample(self.patches, 100)
        for i in range(50):
            patch = self.temp[subpatches[i]]
            pair1, pair2 = self.get4path(patchid=patch)
            print(pair1.shape)
            break



    










def main():
    path = cfg.paths['data']
    temp = datasettemp(iframefolders=path)
   
    dd = VideoNoiseSet(datapath=cfg.paths['iframes'])
    dd[0]
   

if __name__ == "__main__":
    main()