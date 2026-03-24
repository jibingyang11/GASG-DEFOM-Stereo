import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 基础模块 =====
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.Mish(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch)
        )

    def forward(self, x):
        return x + self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg, mx], dim=1)
        return x * self.sigmoid(self.conv(att))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# ===== 增强型生成器 =====
class GeometryAwareStereoGenerator(nn.Module):
    """
    高精度版: 多尺度残差 + 注意力融合 + 深跳跃连接
    输入: 左图 (B,3,H,W)
    输出: 右图 (B,3,H,W)
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(3, 32, 7, 2, 3),
            ResidualBlock(32),
            ChannelAttention(32)
        )
        self.enc2 = nn.Sequential(
            ConvBlock(32, 64, 5, 2, 2),
            ResidualBlock(64),
            ChannelAttention(64)
        )
        self.enc3 = nn.Sequential(
            ConvBlock(64, 128, 3, 2, 1),
            ResidualBlock(128),
            ChannelAttention(128)
        )

        # Bottleneck (深层几何特征)
        self.bottleneck = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            SpatialAttention()
        )

        # Decoder
        self.up1 = UpBlock(128, 64)
        self.dec1 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ResidualBlock(64),
            SpatialAttention()
        )

        self.up2 = UpBlock(64, 32)
        self.dec2 = nn.Sequential(
            ConvBlock(32 + 32, 32),
            ResidualBlock(32),
            ChannelAttention(32)
        )

        self.up3 = UpBlock(32, 16)
        self.dec3 = nn.Sequential(
            ConvBlock(16, 16),
            ResidualBlock(16)
        )

        # 输出层
        self.final_conv = nn.Conv2d(16, 3, 7, padding=3)
    
    def forward(self, left_img):
        # ===== 编码 =====
        x1 = self.enc1(left_img)  # (B,32,H/2,W/2)
        x2 = self.enc2(x1)        # (B,64,H/4,W/4)
        x3 = self.enc3(x2)        # (B,128,H/8,W/8)

        # ===== 几何特征变换 =====
        geo = self.bottleneck(x3)

        # ===== 解码 =====
        y1 = self.up1(geo)
        y1 = torch.cat([y1, x2], dim=1)
        y1 = self.dec1(y1)

        y2 = self.up2(y1)
        y2 = torch.cat([y2, x1], dim=1)
        y2 = self.dec2(y2)

        y3 = self.up3(y2)
        y3 = self.dec3(y3)

        out = torch.sigmoid(self.final_conv(y3)) * 2 - 1
        out = (out + 1) / 2   # 转到 [0, 1]

        # gamma校正：提升亮部但不破坏暗部细节
        gamma = 0.8   # <1 提亮, >1 压暗
        out = out.pow(gamma)
        
        # 可选: 限制亮斑
        out = torch.clamp(out, 0, 1)
        return out
