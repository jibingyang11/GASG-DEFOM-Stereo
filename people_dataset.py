import os
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import torch
class PeopleDataSet(Dataset):
    def __init__(self,root="data/my_own_data/peopledata/train/",have_depth=False,have_disp=False,
                  transforms=T.Compose([T.ToTensor(),T.Resize((256,512)),T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])):
                                        #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])):
        
        self.left_lst=natsorted(os.listdir(root+"left"))
        self.right_lst=natsorted(os.listdir(root+"right"))
        self.depth_lst=None
        self.root=root
        self.baseline=1
        self.force_length=600
        self.have_depth=have_depth
        self.have_disp=have_disp
        if have_depth or have_disp:
            self.depth_lst=natsorted(os.listdir(root+"depth_left_truth"))
            self.depth_transforms=T.Compose([T.ToTensor(),T.Resize((256,512)),T.Normalize(mean=[0.5],std=[0.5])])
        self.transforms=transforms
    def __getitem__(self,idx):
        left_img = Image.open(self.root + "left/" + self.left_lst[idx])
        right_img = Image.open(self.root + "right/" + self.right_lst[idx])
        if self.transforms is not None:
            left_img=self.transforms(left_img)
            right_img=self.transforms(right_img)
        if self.have_depth:
            depth_img = Image.open(self.root + "depth_left_truth/" + self.depth_lst[idx])
            depth_img=np.array(depth_img)[:,:,0]
            depth_img=self.depth_transforms(depth_img)
            #depth_img = torch.clamp(depth_img, 0, 1).float()
        elif self.have_disp:
            depth_img = Image.open(self.root + "depth_left_truth/" + self.depth_lst[idx])
            depth_img=np.array(depth_img)[:,:,0]*(50.0/255.0)
            disp_img=self.depth2disp(depth_img)
            disp_min, disp_max = np.min(disp_img), np.max(disp_img)
            disp_img = (disp_img - disp_min) / (disp_max - disp_min + 1e-8)
            disp_img=self.depth_transforms(disp_img)
        if self.have_depth:
            return left_img,right_img,depth_img
        elif self.have_disp :
            return left_img,right_img,disp_img
        else:
            return left_img,right_img
    def depth2disp(self,depth):
        return (self.baseline*self.force_length)/(depth+1e-8)
    def __len__(self):
        return len(self.left_lst)