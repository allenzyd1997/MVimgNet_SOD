from cgi import test
from lib2to3.pytree import Base
import os
from tabnanny import check
import torch
import sys
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
from PIL import Image,ImageFile
import random 
from pathlib import Path
import einops
from torchvision import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True
class BaseDataset(Dataset):
    def __init__(self, paths, transforms = None):
        super().__init__()
        self.class_name = []
        self.paths = paths
        self.files = self.__getfiles__(self.paths)
        self.file_number = len(self.files)
        self.transforms = transforms

    def __len__(self):
        return self.file_number

    def __getfiles__(self,paths):
        files = [] 
        # 用list储存file， 每一个file保存为[pth， label]
        for path in paths:
            files += self.__getfiles_in_one_path__(path)
        return files 

    def __getfiles_in_one_path__(self, path):
        dirs = os.listdir(path)
        dirs = [d for d in dirs if len(d) < 5]
        dirs = sorted(dirs, key = lambda x: int(x))

        self.class_name  += [x for x in dirs if x not in self.class_name]
        self.class_name = sorted(self.class_name, key = lambda x : int (x))
        
        # the files include [pth, label]
        files = [] 
        for d in dirs :
            pd = os.path.join(path, d)
            subds = os.listdir(pd)
            label = int(d.strip())
            if len(subds) > 0:
                for sd in subds:
                    imgs = os.listdir(os.path.join(pd, sd))
                    if len(imgs) > 0:
                        for img_pth in imgs:
                            img_pth = os.path.join(pd, sd, img_pth)
                            files.append([img_pth,label])
        return files

    def __getitem__(self, idx):
        file = self.files[idx]
        pth , label = file
        img = Image.open(pth)
        if self.transforms:
            img = self.transforms(img)
        # return img, label, pth
        return img,label

class SequenceDataset(BaseDataset):
    def __init__(self, paths, transforms=None):
        self.class_name = []
        self.paths = paths
        self.write_path = '/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles'
        self.train_test_ratio = 0.8
        self.files = self.__getfiles__(self.paths)
        self.file_number = len(self.files)
        self.transforms = transforms 

    def __getfiles_in_one_path__(self, path):
        dirs = os.listdir(path)
        dirs = [d for d in dirs if len(d) < 5]
        dirs = sorted(dirs, key = lambda x: int(x))

        self.class_name  += [x for x in dirs if x not in self.class_name]
        self.class_name = sorted(self.class_name, key = lambda x : int(x))
        # the files include [pth, label]
        files = []
        vds = [] 
        for d in dirs :
            pd = os.path.join(path, d)
            subds = os.listdir(pd)
            label = int(d.strip())
            if len(subds) > 0:
                for sd in subds:
                    if len(os.listdir(os.path.join(pd, sd))) == 30 :
                        vds.append(os.path.join(pd,sd))
        random.shuffle(vds)
        train_len = int(len(vds) * self.train_test_ratio)
        
        filename = Path(self.write_path)/Path("train.lst")
        fw = open(filename, 'w') 
        for vd_p in vds[:train_len]:
            fw.write(vd_p+"\n")
            label = int(Path(vd_p).parent.name.strip())
            for img in sorted(os.listdir(vd_p), key=lambda x:int(x[:3])):
                files.append([os.path.join(vd_p, img), label])

        fw.close()
        filename = Path(self.write_path)/Path("test.lst")
        fw = open(filename, 'w')                
        for vd_p in vds[train_len:]:
            fw.write(vd_p+"\n")
        fw.close()
        return files 



class VideoDataset(BaseDataset):      
    def __init__(self, paths):
        super().__init__(paths)
        self.class_name = []
        self.paths = paths
        self.files = self.__getfiles__(self.paths)
        self.file_number = len(self.files)
        
    def __getfiles_in_one_path__(self, path):
        dirs = os.listdir(path)
        dirs = [d for d in dirs if len(d) < 5]
        dirs = sorted(dirs, key = lambda x: int(x))

        self.class_name  += [x for x in dirs if x not in self.class_name]
        self.class_name = sorted(self.class_name, key = lambda x : int(x))
        # the files include [pth, label]
        files = [] 
        for d in dirs :
            pd = os.path.join(path, d)
            subds = os.listdir(pd)
            label = int(d.strip())
            if len(subds) > 0:
                for sd in subds:
                    imgs = os.listdir(os.path.join(pd, sd))
                    if len(imgs) > 0:
                        files.append([os.path.join(pd, sd), label])
        return files

    def __getitem__(self, idx):
        file = self.files[idx]
        pth , label = file
        # img = Image.open(pth)
        # return img, label, pth
        return pth,label

class VideoTestDataset(Dataset):
    def __init__(self, testLst):
        super().__init__()
        self.files = self.__getfiles__(testLst)
        self.file_number = len(self.files)

    def __getfiles__(self, testfile):
        fr = open(testfile, "r") 
        lst = [] 
        for line in fr.readlines():
            lst.append(line)
        fr.close()
        return lst

    def __len__(self):
        return self.file_number

    def __getitem__(self, idx):
        return self.files[idx]

        

if __name__ == "__main__":
    # pth0 = '/data1/zhangyidan/mvImgNet'
    # pths = [pth0]

    # a_transform = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # ])

    # vd = VideoDataset(paths = pths)
    # files= vd.files
    # print(len(files))
    # # tl = torch.utils.data.DataLoader(td, shuffle = False)
    # # for fi in files :
    #     # print(fi)
    # # print((td.class_name))

    vt = VideoTestDataset('/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles/test.lst')
    print(len(vt))