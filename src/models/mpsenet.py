# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:00:19 2025

@author: gist
"""

import torch
import torch.nn as nn
from .transformer import TransformerBlock


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out
    

class DenseBlock(nn.Module):
    def __init__(self, dim, depth=4, kernel_size=(2, 3)):
        super(DenseBlock, self).__init__()
        
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        
        for i in range(depth):
            
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(dim*(i+1), dim, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(dim, affine=True),
                nn.PReLU(dim)
            )
            
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, dim):
        super(DenseEncoder, self).__init__()

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, (1, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim)
        )

        self.dense_block = DenseBlock(dim, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim)
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, dim, n_fft, beta, out_channel=1):
        super(MaskDecoder, self).__init__()
        
        self.dense_block = DenseBlock(dim, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dim, dim, (1, 3), 2),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.Conv2d(dim, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft//2+1, beta=beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, dim, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(dim, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dim, dim, (1, 3), 2),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim)
        )
        self.phase_conv_r = nn.Conv2d(dim, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dim, out_channel, (1, 2))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x


class TFTransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TFTransformerBlock, self).__init__()
        self.dim = dim
        self.time_transformer = TransformerBlock(d_model=self.dim, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=self.dim, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        
        return x


# for disentanglment

class MagDecoder(nn.Module):
    def __init__(self, dim, n_fft, out_channel=1):
        super(MagDecoder, self).__init__()
        
        self.dense_block = DenseBlock(dim, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dim, dim, (1, 3), 2),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.Conv2d(dim, out_channel, (1, 2))
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]

        return x
