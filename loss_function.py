import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
from pytorch_msssim import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vgg16():
    try:
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        return vgg
    except Exception as e:
        print("加载 VGG16 失败:", e)
        return None


class StereoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.vgg = load_vgg16()

    def perceptual(self, gen, tgt):
        """感知损失"""
        feats_gen, feats_tgt = [], []
        x1, x2 = gen, tgt
        for i, layer in enumerate(self.vgg.features[:23]):
            x1 = layer(x1)
            x2 = layer(x2)
            if i in [3, 8, 15, 22]:
                feats_gen.append(x1)
                feats_tgt.append(x2)
        return sum(self.mse(f1, f2) for f1, f2 in zip(feats_gen, feats_tgt)) / 4

    def color_histogram_loss(self, gen, tgt, bins=64):
        loss = 0
        for c in range(3):
            hist_g = torch.histc(gen[:, c].detach().cpu().clamp(-1, 1), bins=bins, min=-1, max=1)
            hist_t = torch.histc(tgt[:, c].detach().cpu().clamp(-1, 1), bins=bins, min=-1, max=1)
            hist_g /= (hist_g.sum() + 1e-8)
            hist_t /= (hist_t.sum() + 1e-8)
            loss += F.mse_loss(hist_g, hist_t)
        return loss / 3

    def edge_smooth_loss(self, disparity, img):
        """边缘平滑损失"""
        dx = (disparity[:, :, :, :-1] - disparity[:, :, :, 1:]).abs()
        dy = (disparity[:, :, :-1, :] - disparity[:, :, 1:, :]).abs()
        img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        weight_x = torch.exp(-10 * img_grad_x)  # 增强边缘抑制
        weight_y = torch.exp(-10 * img_grad_y)
        return (weight_x * dx).mean() + (weight_y * dy).mean()

    def disparity_loss(self, pre_disparity, gt_disparity):
        if isinstance(pre_disparity, (list, tuple)):
            N = len(pre_disparity)
            loss = 0
            for i, d_i in enumerate(pre_disparity):
                weight = 0.9 ** (N - i - 1)
                loss += weight * F.l1_loss(d_i, gt_disparity)
            return loss
        else:
            return F.l1_loss(pre_disparity, gt_disparity)

    def reconstruction_loss(self, gen, tgt):
        """SSIM + L1"""
        l1 = F.l1_loss(gen, tgt)
        ssim_loss = 1 - ssim(gen, tgt, data_range=1, size_average=True)
        return 0.85 * l1 + 0.15 * ssim_loss

    def gradient_loss(self, gen, tgt):
        grad_gen_x = torch.abs(gen[:, :, :, 1:] - gen[:, :, :, :-1])
        grad_gen_y = torch.abs(gen[:, :, 1:, :] - gen[:, :, :-1, :])
        grad_tgt_x = torch.abs(tgt[:, :, :, 1:] - tgt[:, :, :, :-1])
        grad_tgt_y = torch.abs(tgt[:, :, 1:, :] - tgt[:, :, :-1, :])
        return F.l1_loss(grad_gen_x, grad_tgt_x) + F.l1_loss(grad_gen_y, grad_tgt_y)

    def forward(self, gen, tgt, pre_disparity=None, disparity=None):
        L_rec = self.reconstruction_loss(gen, tgt)

        gen_perc = F.interpolate((gen + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False)
        tgt_perc = F.interpolate((tgt + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False)
        L_perc = self.perceptual(gen_perc, tgt_perc) if self.vgg else 0

        L_color = self.color_histogram_loss(gen, tgt)
        L_grad = self.gradient_loss(gen, tgt)
        L_smooth = self.edge_smooth_loss(disparity, gen) if disparity is not None else 0
        L_disp = self.disparity_loss(pre_disparity, disparity) if pre_disparity is not None and disparity is not None else 0

        total_loss = (0.4 * L_rec +
                      0.1 * L_perc +
                      0.1 * L_color +
                      0.1 * L_grad +
                      0.1 * L_smooth +
                      0.1 * L_disp)

        return total_loss
