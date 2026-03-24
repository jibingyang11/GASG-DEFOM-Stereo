import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
import sys
import torchvision.transforms as T

# 添加路径
sys.path.append('core')
# 导入你的模型
from person_defom_stereo import PersonDEFOMStereo, PredictParameters_PersionDEFOMStereo
from datasets import PersonDEFOMDataset, read_person_defom_filenames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_errors(gt, pred):
    """计算预测深度和真实深度之间的误差指标"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """逆归一化"""
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def evaluate_person_defom():
    parser = argparse.ArgumentParser(description="Evaluate PersonDEFOMStereo on custom dataset")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the custom dataset")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for data loading")
    parser.add_argument("--height", type=int, default=256,
                        help="input image height")
    parser.add_argument("--width", type=int, default=512,
                        help="input image width")
    parser.add_argument("--save_depth_img", type=bool, default=False,
                        help="if save depth image")
    args = parser.parse_args()
    
    # 常量 - 根据你的深度图范围调整
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 50.0  # 你的深度图最大深度为50米
    
    # 创建数据集
    filenames = read_person_defom_filenames(args.data_path)
    dataset = PersonDEFOMDataset(
        args.data_path, 
        filenames, 
        height=args.height,
        width=args.width,
        is_train=False
    )
    
    dataloader = DataLoader(
        dataset, 
        args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=False
    )
    
    # 初始化模型
    print("-> 初始化 PersonDEFOMStereo 模型")
    model = PersonDEFOMStereo()
    
    pred_disps = []
    gt_depths = []
    filenames_list = []
    
    print("-> 在自定义数据集上计算预测，尺寸 {}x{}".format(args.height, args.width))
    print("-> 深度范围: {} - {} 米".format(MIN_DEPTH, MAX_DEPTH))
    
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            lefts = data["left"].to(device)
            rights = data["right"].to(device)
            
            # 使用模型预测视差
            pred_disp = model.forward(lefts,rights=rights)
            
            # 处理预测结果
            if pred_disp.dim() == 4:
                # 如果是4D tensor [B, C, H, W]，取第一个通道
                pred_disp = pred_disp[:, 0, :, :]
            
            pred_disp = pred_disp.cpu().numpy()
            pred_disps.append(pred_disp)
            
            # 保存真实深度（如果有）
            if "depth_gt" in data:
                gt_depths.append(data["depth_gt"].numpy())
            
            # 保存文件名
            filenames_list.extend(data["filename"])
    
    if pred_disps:
        pred_disps = np.concatenate(pred_disps)
    
    # 如果有真实深度，计算指标
    if gt_depths:
        gt_depths = np.concatenate(gt_depths)
        print("-> 计算评估指标")
        print("-> 有效深度样本数: {}".format(len(gt_depths)))
        
        errors = []
        valid_samples = 0
        
        for i in range(len(pred_disps)):
            pred_disp = pred_disps[i]
            gt_depth = gt_depths[i]
            
            # 将视差转换为深度
            # 注意：这里需要根据你的模型输出调整转换公式
            # 假设 pred_disp 已经是视差图，使用公式: depth = (focal_length * baseline) / disp
            focal_length = 600.0  # 根据你的相机调整
            baseline = 1.0       # 根据你的相机调整
            
            pred_depth = (focal_length * baseline) / (pred_disp + 1e-8)
            
            # 创建掩码（有效深度区域）
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            
            # 确保有足够的有效像素
            if np.sum(mask) > 100:  # 至少100个有效像素
                pred_depth_masked = pred_depth[mask]
                gt_depth_masked = gt_depth[mask]
                
                # 应用中值缩放
                ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
                pred_depth_masked *= ratio
                
                # 限制深度范围
                pred_depth_masked = np.clip(pred_depth_masked, MIN_DEPTH, MAX_DEPTH)
                
                errors.append(compute_errors(gt_depth_masked, pred_depth_masked))
                valid_samples += 1
        
        print("-> 有效评估样本: {}/{}".format(valid_samples, len(pred_disps)))
        
        if errors:
            mean_errors = np.array(errors).mean(0)
            print("\n" + "="*60)
            print("PersonDEFOMStereo 评估结果")
            print("="*60)
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            
            # 保存详细结果
            results_file = os.path.join("person_defom_evaluation_results.txt")
            with open(results_file, 'w') as f:
                f.write("PersonDEFOMStereo Evaluation Results\n")
                f.write("=" * 50 + "\n")
                f.write("Dataset: {}\n".format(args.data_path))
                f.write("Samples: {}/{}\n".format(valid_samples, len(pred_disps)))
                f.write("Depth Range: {} - {} meters\n".format(MIN_DEPTH, MAX_DEPTH))
                f.write("Input Size: {}x{}\n".format(args.height, args.width))
                f.write("\nMetrics:\n")
                f.write(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\n")
                f.write(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\\n")
            
            print(f"-> 详细结果保存在: {results_file}")
    
    # 保存预测结果
    if args.save_depth_img:
        output_dir = "person_defom_predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (pred_disp, filename) in enumerate(zip(pred_disps, filenames_list)):
            # 保存视差图
            np.save(os.path.join(output_dir, f"{filename}_disp.npy"), pred_disp)
            
            # 将视差转换为深度并保存
            focal_length = 600.0
            baseline = 1.0
            pred_depth = (focal_length * baseline) / (pred_disp + 1e-8)
            np.save(os.path.join(output_dir, f"{filename}_depth.npy"), pred_depth)
            
            # 保存深度图可视化（0-50米映射到0-255）
            depth_normalized = (pred_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            depth_img = (depth_normalized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_depth.png"), depth_img)
            
            # 保存视差图可视化
            disp_normalized = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
            disp_img = (disp_normalized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_disp.png"), disp_img)
        
        print(f"-> 完成！预测结果保存在 {output_dir}")

if __name__ == "__main__":
    evaluate_person_defom()