import torch
from people_dataset import *
from model import *
from torchinfo import summary
from PIL import Image
import torchvision.transforms as T
import numpy as np
from people_dataset import *
import sys
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from matplotlib import pyplot as plt
import json
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PersonDEFOMStereo():
    def __init__(self):
        self.predict_parser=PredictParameters_PersionDEFOMStereo()
        self.args=self.predict_parser.parse()
        self.model_predict_right=GeometryAwareStereoGenerator()
        self.model_predict_disp=DEFOMStereo(self.args)
        self.model_predict_right.load_state_dict(torch.load('models/right_generator/my_right_generator_GAN_5.pth',map_location=device,weights_only=True ))
        self.model_predict_disp.load_state_dict(torch.load('models/defom_stereo/defomstereo_vitl_kitti.pth',map_location=device,weights_only=True ))
        with torch.no_grad():
            self.model_predict_right.eval()
            self.model_predict_disp.eval()
    def forward(self,lefts,rights=None):
        disps=predict_disp(lefts,self.model_predict_right,self.model_predict_disp,self.args,rights=rights)
        return disps
    def eval(self):
        self.model_predict_right.eval()
        self.model_predict_disp.eval()
    def to(self,device):
        self.model_predict_right=self.model_predict_right.to(device)
        self.model_predict_disp=self.model_predict_disp.to(device)
class PredictParameters_PersionDEFOMStereo():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
        self.parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")
        # DefomStereo Architecture choices
        self.parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
        self.parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
        self.parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        self.parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        self.parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        self.parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        self.parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        self.parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                            help='the list of scaling factors of disparity')
        self.parser.add_argument('--scale_corr_radius', type=int, default=2,
                            help="width of the correlation pyramid for scaled disparity")
        self.parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
        self.parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
        self.parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    def parse(self):
        self.args,_=self.parser.parse_known_args()
        return self.args
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)  
    tensor = (tensor * 255).float()
    return tensor
def predict_rights(lefts,model):
    predict_right=model(lefts).detach()
    predict_right = predict_right
    predict_right=predict_right.to(device)
    return predict_right
def predict_disp(lefts,model_predict_right,model_predict_disp,args,rights=None):
    lefts=lefts.to(device)
    if rights==None:
        rights=predict_rights(lefts,model_predict_right)
    rights=rights.to(device)
    padder = InputPadder(lefts.shape, divis_by=32)
    lefts, rights = padder.pad(lefts, rights)
    lefts=denormalize(lefts)
    rights=denormalize(rights,mean=[0],std=[1])
    with torch.no_grad():
        disp=model_predict_disp(lefts, rights, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True).detach()
    return disp
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]