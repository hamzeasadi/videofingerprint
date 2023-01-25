import cv2
import os
import conf as cfg



def iframeextractionvideo(videopath, trgpath):
    # command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync 0 -frame_pts true {filepath}out%d.png"
    nn = cfg.paths['data']
    srcfolders = os.listdir(videopath)
    srcfolders = cfg.ds_rm(srcfolders)
    for srcfolder in srcfolders:
        trgfolder = os.path.join(trgpath, srcfolder)
        srcfolderpath = os.path.join(videopath, srcfolder)
        cfg.creatdir(trgfolder)
        videos = os.listdir(srcfolderpath)
        videos = cfg.ds_rm(videos)
        for i, video in enumerate(videos):
            videopathfile = os.path.join(srcfolderpath, video)
            command = f"ffmpeg -skip_frame nokey -i {videopathfile} -vsync vfr -frame_pts true -x264opts no-deblock {trgfolder}/video{i}out%d.bmp"
            os.system(command=command)








def main():
    # path = '/Volumes/myDrive/Datasets/visionDataset copy/D01_Samsung_GalaxyS3Mini/D01_V_flat_move_0001.mp4'
    # path = 'D09_V_indoor_move_0001.mov'
    # path = 'D01_V_flat_move_0001.mp4'
    # iframeextractionvideo(videopath=path)
    srcpath = cfg.paths['videos']
    trgpath = cfg.paths['iframes']
    iframeextractionvideo(videopath=srcpath, trgpath=trgpath)

if __name__ == '__main__':
    main()