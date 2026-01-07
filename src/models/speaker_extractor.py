# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 21:30:48 2025

@author: gist
"""

# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py


import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.nn.functional as F

#from ptflops import get_model_complexity_info

class ConvNeXtV2(nn.Module):    
    def __init__(self, emb_dim, depths = [2, 2, 3, 2]):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.depths = depths
        self.num_layers = len(depths)
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=4, stride=4),
            LayerNorm_(emb_dim, eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(self.num_layers-1):
            downsample_layers = nn.Sequential(
                LayerNorm_(emb_dim, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(emb_dim, emb_dim, kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layers)
        
        self.stages = nn.ModuleList()
        
        for i in range(self.num_layers):
            stage = nn.Sequential(
                *[Block(dim=self.emb_dim) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            
        
        self.TAP = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm_(emb_dim, eps=1e-6)
        
        self.fc  = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, auxs):
        
        for i in range(self.num_layers):
            
            auxs = self.downsample_layers[i](auxs)
            auxs = self.stages[i](auxs)  # [B, C, T, F]
            #print('{}-th, speaker_encoder: {}'.format(i, auxs.shape))
            
        auxs = self.TAP(auxs).view(auxs.size(0), -1)
        auxs = self.norm(auxs)
        
        spk_embedding = self.fc(auxs)
        
        return spk_embedding

class UNet(nn.Module):
    def __init__(self, emb_dim):
        super(UNet, self).__init__()
        k1, k2 = (1, 3), (1, 3)
        
        self.emb_dim = emb_dim

        self.aux_enc = nn.ModuleList([EnUnetModule(emb_dim, emb_dim, (1, 5), k2, scale=4),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=3),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=2),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=1)])
    

        self.TAP = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, auxs):
        #print(auxs.shape)
        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [B, C, T, F]
            #print('{}-th, speaker_encoder: {}'.format(i, auxs.shape))
     
        auxs = self.TAP(auxs).squeeze(dim=-2).squeeze(dim=-1)
        
        spk_embedding = self.fc(auxs)
        

        return spk_embedding

class EnUnetModule(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 scale: int):
        super(EnUnetModule, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.scale = scale

        self.in_conv = nn.Sequential(GateConv2d(cin, cout, k1, (1, 2)),
                                     nn.BatchNorm2d(cout),
                                     nn.PReLU(cout))
        self.encoder = nn.ModuleList([Conv2dUnit(k2, cout) for _ in range(scale)])
        self.decoder = nn.ModuleList([Deconv2dUnit(k2, cout, 1)])
        for i in range(1, scale):
            self.decoder.append(Deconv2dUnit(k2, cout, 2))
        self.out_pool = nn.AvgPool2d((3, 1))

    def forward(self, x: torch.Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)

        x = self.decoder[0](x)
        for i in range(1, len(self.decoder)):
            x = self.decoder[i](torch.cat([x, x_list[-(i + 1)]], dim=1))
        x_resi = x_resi + x

        return self.out_pool(x_resi)


class GateConv2d(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k: tuple,
                 s: tuple):
        super(GateConv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s

        self.conv = nn.Sequential(nn.ConstantPad2d((0, 0, k[0] - 1, 0), value=0.),
                                  nn.Conv2d(in_channels=cin,
                                            out_channels=cout * 2,
                                            kernel_size=k,
                                            stride=s))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)

        return outputs * gate.sigmoid()


class Conv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int):
        super(Conv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.conv = nn.Sequential(nn.Conv2d(c, c, k, (1, 2)),
                                  nn.BatchNorm2d(c),
                                  nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)

class Deconv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 expend_scale: int):
        super(Deconv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.expend_scale = expend_scale
        self.deconv = nn.Sequential(nn.ConvTranspose2d(c * expend_scale, c, k, (1, 2)),
                                    nn.BatchNorm2d(c),
                                    nn.PReLU(c))

    def forward(self, x):
        return self.deconv(x)
    
    
    
class LayerNorm_(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm_(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        
        return x
    
if __name__ == '__main__':
    
    device = torch.device("cuda")
    
    emb_dimension = 40
    
    model1 = UNet(emb_dim=emb_dimension).to(device)
    model2 = ConvNeXtV2(emb_dim=emb_dimension).to(device)
    
    def prepare_input(input_shape):
        
        inputs = {
            'auxs': torch.ones((2, *input_shape)).to(device),
        }
        
        return inputs
    
    # 4-seconds: 126
    # 10-seconds: 313
    #
    frames = 126
    
    macs1, params1 = get_model_complexity_info(model1,  (emb_dimension, frames, 65), as_strings=True, backend='pytorch',
                                     print_per_layer_stat=True, verbose=True, input_constructor = prepare_input)
    
    macs2, params2 = get_model_complexity_info(model2,  (emb_dimension, frames, 65), as_strings=True, backend='pytorch',
                                     print_per_layer_stat=True, verbose=True, input_constructor = prepare_input)
    
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    print('Computational complexity || U^2-Net: {}, ConvNeXt-V2: {}'.format(macs1, macs2))
    print('Number of parameters || U^2-Net: {}, ConvNeXt-V2: {}'.format(params1, params2))
    
    
    
    
    