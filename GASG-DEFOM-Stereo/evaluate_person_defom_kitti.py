# evaluate_depth_selection.py
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
import matplotlib.pyplot as plt

sys.path.append('core')
from person_defom_stereo import PersonDEFOMStereo
from kitti_dataset import KITTIDepthSelectionDataset

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

def visualize_results(image, pred_depth, gt_depth, filename, save_dir):
    """可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 预测深度
    im1 = axes[1].imshow(pred_depth, cmap='plasma', vmin=0, vmax=80)
    axes[1].set_title('Predicted Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # 真实深度
    if gt_depth.max() > 0:  # 如果有真值
        im2 = axes[2].imshow(gt_depth, cmap='plasma', vmin=0, vmax=80)
        axes[2].set_title('Ground Truth Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, 'No GT Available', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Ground Truth Depth')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_depth_selection():
    parser = argparse.ArgumentParser(description="Evaluate on KITTI Depth Selection")
    
    parser.add_argument("--data_path", type=str, default="data/kitti_eidge",
                        help="path to the KITTI dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="split to evaluate on: 'val' or 'test'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for evaluation")
    parser.add_argument("--height", type=int, default=352,
                        help="input image height")
    parser.add_argument("--width", type=int, default=1216,
                        help="input image width")
    parser.add_argument("--visualize", action="store_true",
                        help="whether to visualize results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="whether to save prediction files")
    
    args = parser.parse_args()
    
    # KITTI 参数
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80.0
    FOCAL_LENGTH = 721.5377
    BASELINE = 0.54
    
    # 创建数据集
    dataset = KITTIDepthSelectionDataset(
        data_path=args.data_path,
        split=args.split,
        height=args.height,
        width=args.width
    )
    
    dataloader = DataLoader(
        dataset, 
        args.batch_size, 
        shuffle=False, 
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True
    )
    
    # 初始化模型
    print("-> 初始化 PersonDEFOMStereo 模型")
    model = PersonDEFOMStereo()
    model.eval()
    model.to(device)
    
    print(f"-> 在KITTI Depth Selection {args.split}集上评估")
    print(f"-> 样本数量: {len(dataset)}")
    print(f"-> 输入尺寸: {args.height}x{args.width}")
    
    # 存储结果
    all_errors = []
    all_filenames = []
    all_pred_depths = []
    all_gt_depths = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Processing"):
            images = batch["image"].to(device)
            gt_depths = batch["depth_gt"].numpy()
            filenames = batch["filename"]
            
            # 模型预测
            pred_disps = model.forward(images)
            
            # 处理预测结果
            if pred_disps.dim() == 4:
                pred_disps = pred_disps[:, 0, :, :]  # 取第一个通道
            
            pred_disps = pred_disps.cpu().numpy()
            
            for i in range(len(pred_disps)):
                pred_disp = pred_disps[i]
                gt_depth = gt_depths[i]
                filename = filenames[i]
                
                # 视差转深度
                pred_depth = (FOCAL_LENGTH * BASELINE) / (pred_disp + 1e-8)
                
                # 存储结果
                all_filenames.append(filename)
                all_pred_depths.append(pred_depth)
                all_gt_depths.append(gt_depth)
                
                # 如果有真值且是验证集，计算误差
                if args.split == 'val' and gt_depth.max() > 0:
                    # 创建有效掩码
                    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                    
                    if np.sum(mask) > 0:
                        pred_depth_masked = pred_depth[mask]
                        gt_depth_masked = gt_depth[mask]
                        
                        # KITTI标准：应用中值缩放
                        ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
                        pred_depth_masked *= ratio
                        
                        # 限制深度范围
                        pred_depth_masked = np.clip(pred_depth_masked, MIN_DEPTH, MAX_DEPTH)
                        gt_depth_masked = np.clip(gt_depth_masked, MIN_DEPTH, MAX_DEPTH)
                        
                        errors = compute_errors(gt_depth_masked, pred_depth_masked)
                        all_errors.append(errors)
                
                # 可视化
                if args.visualize and i == 0:  # 只可视化每个batch的第一个
                    image_np = images[i].cpu().permute(1, 2, 0).numpy()
                    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                    visualize_results(image_np, pred_depth, gt_depth, filename, "visualizations")
    
    # 输出结果
    if args.split == 'val' and all_errors:
        mean_errors = np.array(all_errors).mean(0)
        
        print("\n" + "="*80)
        print(f"PersonDEFOMStereo 在 KITTI Depth Selection {args.split}集上的评估结果")
        print("="*80)
        print(f"有效样本: {len(all_errors)}/{len(dataset)}")
        print("\n  " + ("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 10.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        
        # 保存详细结果
        results_file = f"kitti_depth_selection_{args.split}_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"PersonDEFOMStereo KITTI Depth Selection {args.split} Results\n")
            f.write("="*50 + "\n")
            f.write(f"数据路径: {args.data_path}\n")
            f.write(f"有效样本: {len(all_errors)}/{len(dataset)}\n")
            f.write(f"输入尺寸: {args.height}x{args.width}\n\n")
            f.write("指标结果:\n")
            f.write(("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\n")
            f.write(("&{: 10.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\\n")
        
        print(f"-> 详细结果保存至: {results_file}")
    
    # 保存预测结果
    if args.save_predictions:
        output_dir = f"predictions_{args.split}"
        os.makedirs(output_dir, exist_ok=True)
        
        for filename, pred_depth in zip(all_filenames, all_pred_depths):
            # 保存numpy文件
            np.save(os.path.join(output_dir, f"{filename}_depth.npy"), pred_depth)
            
            # 保存可视化深度图
            depth_normalized = (pred_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            depth_img = (depth_normalized * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_PLASMA)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_depth.png"), depth_colored)
        
        print(f"-> 预测结果保存至: {output_dir}")
    
    print("-> 评估完成!")

if __name__ == "__main__":
    evaluate_depth_selection()