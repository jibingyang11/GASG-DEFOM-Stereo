import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2

class PersonDEFOMDataset(data.Dataset):
    """自定义数据集类，适配PersonDEFOMStereo模型"""
    
    def __init__(self, data_path, filenames, height=256, width=512, is_train=False):
        super(PersonDEFOMDataset, self).__init__()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.max_depth = 50.0  # 你的深度图最大深度为50米
        
        # 图像预处理transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 深度图transform（只做ToTensor和Resize）
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((height, width))
        ])
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        """获取数据项"""
        inputs = {}
        
        # 获取基础文件名（去掉扩展名）
        base_filename = self.filenames[index].replace('.png', '')
        
        # 加载左图像
        left_path = os.path.join(self.data_path, 'left', f"{base_filename}.png")
        left_img = Image.open(left_path).convert('RGB')
        left_tensor = self.transform(left_img)
        
        # 加载右图像
        right_path = os.path.join(self.data_path, 'right', f"{base_filename}.png")
        right_img = Image.open(right_path).convert('RGB')
        right_tensor = self.transform(right_img)
        
        # 加载真实深度图并转换为米
        depth_path = os.path.join(self.data_path, 'depth_left_truth', f"depth{base_filename}.png")
        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path)
            depth_array = np.array(depth_img)
            
            # 处理深度图：0-255 转换为 0-50米
            if len(depth_array.shape) == 3:
                depth_array = depth_array[:, :, 0]
            
            # 将0-255转换为0-50米
            depth_in_meters = (depth_array / 255.0) * self.max_depth
            
            # 调整深度图尺寸到模型输入尺寸
            depth_resized = cv2.resize(depth_in_meters, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            
            inputs["depth_gt"] = torch.from_numpy(depth_resized.astype(np.float32))
        
        inputs["left"] = left_tensor
        inputs["right"] = right_tensor
        inputs["filename"] = base_filename
        
        return inputs

def read_person_defom_filenames(data_path):
    """读取自定义数据集的文件名"""
    left_dir = os.path.join(data_path, 'left')
    filenames = [f for f in sorted(os.listdir(left_dir)) if f.endswith('.png')]
    return filenames