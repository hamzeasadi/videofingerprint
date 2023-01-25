import os




root = os.pardir
data = os.path.join(root, 'data')

paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'), 
    videos=os.path.join(data, 'videos'), iframes=os.path.join(data, 'iframes'),
    paths=os.path.join(data, 'patches'),
    srcvideospath='/Volumes/myDrive/Datasets/visionDataset copy',
    srciframespath='/Volumes/myDrive/Datasets/visionDatasetnonstbl/iframes',
    srcpathces='/Volumes/myDrive/Datasets/visionDatasetnonstbl/patches',
    srcallpathces='/Volumes/myDrive/Datasets/visionDatasetnonstbl/allpatches',

)


def ds_rm(array: list):
    try:
        array.remove('DS_Store')
    except Exception as e:
        print(e)

def creatdir(path):
    try:
        os.makedirs(path)
    except Exception as ex:
        print(ex)



def main():
    pass



if __name__ == '__main__':
    main()