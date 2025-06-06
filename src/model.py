import torch
import torch.nn as nn
import logging
import os

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if downsample else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)  # Pool first
        skip = x  # Skip connection after pooling
        return x, skip

class SimpleUNet(nn.Module):
    def __init__(self, base_channels=64):
        super(SimpleUNet, self).__init__()
        self.down1 = UNetBlock(1, base_channels)    # base_channels x 32 x 16
        self.down2 = UNetBlock(base_channels, base_channels * 2)  # (base_channels * 2) x 16 x 8
        self.down3 = UNetBlock(base_channels * 2, base_channels * 4) # (base_channels * 4) x 8 x 4
        self.bottleneck = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.up1 = UNetBlock(base_channels * 4, base_channels * 2, downsample=False)      # (base_channels * 2) x 16 x 8
        self.up2 = UNetBlock(base_channels * 4, base_channels, downsample=False) # base_channels x 32 x 16
        self.up3 = UNetBlock(base_channels * 2, base_channels, downsample=False)   # base_channels x 64 x 32
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x, t):
        down1, skip1 = self.down1(x)      # skip1: base_channels x 32 x 16
        down2, skip2 = self.down2(down1)  # skip2: (base_channels * 2) x 16 x 8
        down3, skip3 = self.down3(down2)  # skip3: (base_channels * 4) x 8 x 4
        bottle = self.bottleneck(down3)   # (base_channels * 4) x 8 x 4
        up1, _ = self.up1(bottle)         # (base_channels * 2) x 16 x 8
        up2, _ = self.up2(torch.cat([up1, skip2], dim=1))  # (base_channels * 4) x 16 x 8 -> base_channels x 32 x 16
        up3, _ = self.up3(torch.cat([up2, skip1], dim=1))  # (base_channels * 2) x 32 x 16 -> base_channels x 64 x 32
        out = self.out(up3)               # 1 x 64 x 32
        return out, skip3

class SimpleScheduler:
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.device = device

    def add_noise(self, x, noise, t):
        sqrt_alpha = torch.sqrt(1 - self.betas[t]).view(-1, 1, 1, 1)
        sigma = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)
        return sqrt_alpha * x + sigma * noise

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128, dropout_rate=0.3):
        super(ClassificationHead, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(128, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv_reduce(x)  # [batch_size, 128, 8, 4]
        x = self.pool(x)         # [batch_size, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 128]
        x = self.fc2(x)          # [batch_size, hidden_dim]
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc_out(x)       # [batch_size, num_classes]
        return x

class DiffusionClassifier(nn.Module):
    def __init__(self, num_classes, base_channels=64, classifier_hidden_dim=128, dropout_rate=0.3):
        super(DiffusionClassifier, self).__init__()
        self.unet = SimpleUNet(base_channels=base_channels)
        self.classifier = ClassificationHead(
            in_channels=base_channels * 4,  # skip3 has base_channels * 4 channels
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, x, t):
        unet_out, features = self.unet(x, t)
        logits = self.classifier(features)
        return logits, unet_out  # Return both logits and denoising output