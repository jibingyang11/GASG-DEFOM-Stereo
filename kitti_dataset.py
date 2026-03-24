# kitti_depth_selection_dataset.py
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T

class KITTIDepthSelectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='val', height=352, width=1216):
        """
        KITTI Depth Selection数据集
        split: 'val' 或 'test'
        """
        self.data_path = data_path
        self.height = height
        self.width = width
        self.split = split
        
        # 根据split选择目录
        if split == 'val':
            self.image_dir = os.path.join(data_path, "depth_selection", "val_selection_cropped", "image")
            self.depth_dir = os.path.join(data_path, "depth_selection", "val_selection_cropped", "groundtruth_depth")
        elif split == 'test':
            self.image_dir = os.path.join(data_path, "depth_selection", "test_depth_prediction_anonymous", "image")
            self.depth_dir = None  # 测试集没有深度真值
        
        # 获取所有图像文件
        self.image_files = self.get_image_files()
        
        # 预处理
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        
        print(f"加载KITTI Depth Selection {split}集: {len(self.image_files)} 个样本")
    
    def get_image_files(self):
        """获取所有图像文件"""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        
        image_files = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                image_files.append(filename)
        
        return sorted(image_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def get_depth_filename(self, image_filename):
        """根据图像文件名生成对应的深度图文件名"""
        # 图像文件名示例: 2011_09_26_drive_0002_sync_image_0000000005_image_02.png
        # 深度图文件名示例: 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
        
        # 方法1: 直接替换第一个 'image' 为 'groundtruth_depth'
        parts = image_filename.split('_')
        
        # 找到第一个 'image' 的位置
        try:
            image_index = parts.index('image')
            # 替换为 'groundtruth_depth'
            parts[image_index] = 'groundtruth_depth'
            depth_filename = '_'.join(parts)
            return depth_filename
        except ValueError:
            # 如果找不到 'image'，尝试其他方法
            pass
        
        # 方法2: 如果方法1失败，使用简单的字符串替换
        depth_filename = image_filename.replace('image_', 'groundtruth_depth_', 1)
        return depth_filename
    
    def __getitem__(self, index):
        filename = self.image_files[index]
        frame_id = filename.replace('.png', '')
        
        # 加载RGB图像
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸用于后处理
        original_height, original_width = image.shape[:2]
        
        # 调整尺寸
        image = cv2.resize(image, (self.width, self.height))
        
        # 加载深度图真值（如果有）
        depth_gt = np.zeros((self.height, self.width), dtype=np.float32)
        
        if self.split == 'val' and self.depth_dir:
            # 使用新的文件名转换方法
            depth_filename = self.get_depth_filename(filename)
            depth_path = os.path.join(self.depth_dir, depth_filename)
            
            if os.path.exists(depth_path):
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_gt is not None:
                    depth_gt = depth_gt.astype(np.float32) / 256.0  # KITTI深度图格式
                    depth_gt = cv2.resize(depth_gt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                else:
                    print(f"警告: 无法读取深度图 {depth_path}")
            else:
                print(f"警告: 深度图文件不存在 {depth_path}")
        
        # 预处理图像
        image_tensor = self.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        
        return {
            "image": image_tensor,
            "depth_gt": torch.from_numpy(depth_gt),
            "filename": frame_id,
            "original_size": (original_height, original_width)
        }
    
    def get_camera_intrinsics(self, filename):
        """获取相机内参"""
        # 修正内参文件路径
        base_filename = filename.replace('.png', '')
        
        if self.split == 'val':
            intrinsics_dir = os.path.join(self.data_path, "depth_selection", "val_selection_cropped", "intrinsics")
        else:
            intrinsics_dir = os.path.join(self.data_path, "depth_selection", "test_depth_prediction_anonymous", "intrinsics")
        
        intrinsics_path = os.path.join(intrinsics_dir, f"{base_filename}.txt")
        
        if os.path.exists(intrinsics_path):
            try:
                with open(intrinsics_path, 'r') as f:
                    lines = f.readlines()
                    # KITTI内参矩阵通常是 3x3
                    intrinsics = np.array([list(map(float, line.split())) for line in lines])
                    return intrinsics
            except Exception as e:
                print(f"警告: 无法读取内参文件 {intrinsics_path}: {e}")
        
        # 默认KITTI内参 (根据您的图像尺寸调整)
        fx, fy = 721.5377, 721.5377
        cx, cy = 596.5593, 149.854
        
        # 根据调整后的尺寸缩放内参
        scale_x = self.width / original_width
        scale_y = self.height / original_height
        
        return np.array([
            [fx * scale_x, 0, cx * scale_x],
            [0, fy * scale_y, cy * scale_y],
            [0, 0, 1]
        ])