# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:12:25 2025

@author: gist
"""

import sys
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .mpsenet import DenseEncoder, MaskDecoder, PhaseDecoder, TFTransformerBlock, MagDecoder
from .ccblock import CrossCorrelationBlock
from .speaker_extractor    import ConvNeXtV2, UNet

def _inverse_power_compression_and_istft(maginitude, phase, length, compression_factor = None, device = None, sample_rate = 8000, n_fft = 256, hop_size = None, win_length = None):
    
    if win_length is None:
        win_length = n_fft
    if hop_size is None:
        hop_size = n_fft // 2
    
    # Apply the un-compression
    if compression_factor is not None:
        maginitude = maginitude ** (1.0/compression_factor)
    
    # Reconsturct the original real and image compoments using power-uncompression
    original_real = maginitude * torch.cos(phase)
    original_imag = maginitude * torch.sin(phase)
    
    stft_result = torch.stack([original_real, original_imag], dim=-1) # [batch, F, T, 2]
    stft_result_to_complex = torch.view_as_complex(stft_result)
    
    waveform = torch.istft(
        input           = stft_result_to_complex,
        n_fft           = n_fft,
        hop_length      = hop_size,
        win_length      = win_length,
        window          = torch.hann_window(win_length).to(device),
        length          = length
    ) # [Batch, length]
    
    # [batch, length] -> [batch, 1, length]
    waveform = waveform.unsqueeze(dim=1) 
    
    return waveform

def _stft_and_power_compression(audio, compression_factor = None, device = None, sample_rate = 8000, n_fft = 256, hop_size = None, win_length = None):
    
    if win_length is None:
        win_length = n_fft
    if hop_size is None:
        hop_size = n_fft // 2
    
    audio = audio.squeeze(dim=1) # [batch, 1, 32000] -> [batch, 32000]
    
    stft_result = torch.stft(
        audio,
        n_fft           = n_fft,
        hop_length      = hop_size,
        win_length      = win_length,
        window          = torch.hann_window(win_length).to(device),
        return_complex  = True
    )
    
    # Separate real and imaginary parts
    real = stft_result.real
    imag = stft_result.imag
    
    # Compute magnitude and phase
    magnitude   = torch.sqrt(real**2 + imag**2)
    phase       = torch.atan2(imag, real)
    
    # Apply the dyanamic compression
    if compression_factor is not None:
        magnitude = magnitude ** compression_factor
    
    return magnitude, phase


class CrossMultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 nhead=4,
                 dropout=0.1):
        super(CrossMultiHeadAttentionBlock, self).__init__()
        self.nhead = nhead
        self.dropout = dropout

        self.emb_dim = emb_dim

        param_size = [1, 1, emb_dim]
        
        self.attention = nn.MultiheadAttention(emb_dim,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        
        self.fusion = nn.Conv2d(2*emb_dim, emb_dim, kernel_size=1, bias=False)
        
        
        self.alpha = Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.alpha)
    
    def forward(self, x, z):
        
        """
            x: [B, C, T, F]
            z: [B, C]
        """
        z = z.unsqueeze(dim=1)  # [B, 1, C]
        
        x_flatten = x.flatten(start_dim=2).transpose(1, 2)  # [B, C, T, F] -> [B, C, T*F] -> [B, T*F, C]
        z_attn, _ = self.attention(z, x_flatten, x_flatten, need_weights=False)
        
        z = z + self.alpha*z_attn  
        z = z.unsqueeze(-1).transpose(1, 2).expand_as(x)

        output = self.fusion(torch.cat((x, z), dim=1))  # [B, C, T, F]
        
        return output

class CROCS_Coarse(nn.Module):
    def __init__(self, device, **kwargs):
        super(CROCS_Coarse, self).__init__()
        
        """
        Args:
            R: Number of repeats = the number of basic blocks # 6
            M: Number of attention heads # 4 
            H: Number of hidden units in LSTM & Number of chnnels size of encoder # 64
        """
        
        self.device = device
        
        self.R = kwargs['R_coarse']
        self.M = kwargs['M']
        self.H = kwargs['H']
        self.fusion_type = kwargs['fusion_type']
        self.temperature = kwargs['temperature']
        self.compression_factor = kwargs['compression_factor']
        
        self.n_fft          = kwargs['n_fft']
        self.window_length  = kwargs['window_length']
        self.hop_size       = kwargs['hop_size']
        
        self.beta  = 2.0
        
        #print('coarse-stage: -- n_fft: {}, window_length: {}, hop_size:{}'.format(self.n_fft, self.window_length, self.hop_size))
        print('## coarse-stage ||| fusion-type: {}, n_fft: {}, window_length: {}, hop_size: {}'.format(self.fusion_type, self.n_fft, self.window_length, self.hop_size))
        
        
        self.fusion_block  = CrossCorrelationBlock(fusion_type = self.fusion_type, T = self.temperature)
        
        self.dense_encoder = DenseEncoder(in_channel=4, dim = self.H)
        
        self.TFTransformer = nn.ModuleList([])
        
        for i in range(self.R):
            
            self.TFTransformer.append(TFTransformerBlock(dim = self.H))

        self.mask_decoder  = MaskDecoder(dim = self.H, n_fft = self.n_fft, beta = self.beta, out_channel = 1)
        self.phase_decoder = PhaseDecoder(dim = self.H, out_channel=1)
        
        
    def forward(self, mixture, reference): 
        
        length = mixture.shape[-1]
        
        # mix_mag, mix_pha, ref_mga, ref_pha: [batch, F, T_mix or T_ref]
        mix_mag, mix_pha = _stft_and_power_compression(mixture, self.compression_factor, self.device, n_fft=self.n_fft, hop_size=self.hop_size, win_length=self.window_length)
        ref_mag, ref_pha = _stft_and_power_compression(reference, self.compression_factor, self.device, n_fft=self.n_fft, hop_size=self.hop_size, win_length=self.window_length)
        
    
        fus_mag, fus_pha = self.fusion_block(mix_mag, mix_pha, ref_mag, ref_pha)
        x = torch.stack([mix_mag, mix_pha, fus_mag, fus_pha],dim=1) # torch.Size([3, 4, 129, 32])
        x = x.transpose(3, 2) 
        x = self.dense_encoder(x) # [batch, 4, T, F] --> [batch, H, T, F/2]
        
        ref_mag_pha = torch.stack([ref_mag, ref_pha], dim = 1)
        
        for i in range(self.R):
           
            x = self.TFTransformer[i](x)

        mask = self.mask_decoder(x) # [batch, F, T_mix]
        
        est_mag = mix_mag * mask
        est_pha = self.phase_decoder(x)
        
        estimated = _inverse_power_compression_and_istft(est_mag, est_pha, length, self.compression_factor, self.device, n_fft=self.n_fft, hop_size=self.hop_size, win_length=self.window_length)
        
        return estimated, mix_mag, mix_pha, est_mag, est_pha, ref_mag_pha


class CROCS_Fine(nn.Module):
    def __init__(self, device, **kwargs):
        super(CROCS_Fine, self).__init__()
        
        """
        Args:
            R: Number of repeats = the number of basic blocks # 6
            M: Number of attention heads # 4 
            H: Number of hidden units in LSTM & Number of chnnels size of encoder # 64
        """
        
        self.device = device
        
        self.R = kwargs['R_fine']
        
        self.M = kwargs['M']
        self.H = kwargs['H']
        
        self.fusion_type = kwargs['fusion_type']
        self.temperature = kwargs['temperature']
        
        self.fusion_block  = CrossCorrelationBlock(fusion_type = self.fusion_type, T = self.temperature)
        
        self.n_fft          = kwargs['n_fft']
        self.window_length  = kwargs['window_length']
        self.hop_size       = kwargs['hop_size']
        self.beta  = 2.0
        
        print('## fine-stage ||| fusion-type: {}, n_fft: {}, window_length: {}, hop_size: {}'.format(self.fusion_type, self.n_fft, self.window_length, self.hop_size))
        
        self.dense_encoder1 = nn.Sequential(
            DenseEncoder(in_channel = 4, dim = self.H),
            nn.GroupNorm(1, self.H)
        )
        self.dense_encoder2 = nn.Sequential(
            DenseEncoder(in_channel = 4, dim = self.H),
            nn.GroupNorm(1, self.H)
        )
   
        self.TFTransformer1 = nn.ModuleList([])
        self.FusionAdapter1 = nn.ModuleList([])
        
        self.TFTransformer2 = nn.ModuleList([])
        self.FusionAdapter2 = nn.ModuleList([])
        
        # Adapter
        for i in range(self.R):
            
            self.TFTransformer1.append(TFTransformerBlock(dim = self.H))
            self.FusionAdapter1.append(CrossMultiHeadAttentionBlock(emb_dim = self.H))
   
            self.TFTransformer2.append(TFTransformerBlock(dim = self.H))
            self.FusionAdapter2.append(CrossMultiHeadAttentionBlock(emb_dim = self.H))

        self.mask_decoder1  = MaskDecoder(dim = self.H, n_fft = self.n_fft, beta = self.beta, out_channel = 1)
        self.phase_decoder1 = PhaseDecoder(dim = self.H, out_channel=1)
        
        self.mask_decoder2  = MaskDecoder(dim = self.H, n_fft = self.n_fft, beta = self.beta, out_channel = 1)
        self.phase_decoder2 = PhaseDecoder(dim = self.H, out_channel=1)
        
        #self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, mix_mag, mix_pha, est_mag, est_pha, ref_mag, ref_pha, z_ref, z_est): 
        
        
        # mix, est
        x1 = torch.stack([mix_mag, mix_pha, est_mag, est_pha],dim=1)
        x1 = x1.transpose(3, 2)
        
        x1 = self.dense_encoder1(x1)
        
        z = torch.cat([z_ref, z_est], dim = -1)
        
        for i in range(self.R):
            
            x1 = self.TFTransformer1[i](x1)
            x1 = self.FusionAdapter1[i](x1, z)

        mask1 = self.mask_decoder1(x1) # [batch, F, T_mix]
        
        # # estimated using both 
        est2_mix_est_mag = mix_mag * mask1
        est2_mix_est_pha = self.phase_decoder1(x1)
        
        # # ------------- parallel
        fus_mag, fus_pha = self.fusion_block(est_mag, est_pha, ref_mag, ref_pha)
        x2 = torch.stack([est_mag, est_pha, fus_mag, fus_pha], dim=1)
        x2 = x2.transpose(3, 2)
        
        x2 = self.dense_encoder2(x2)
        
        for i in range(self.R):
            
            x2 = self.TFTransformer2[i](x2)
            x2 = self.FusionAdapter2[i](x2, z)
            
        # # existing method
        mask2 = self.mask_decoder2(x2)
        est2_est_ref_mag = est_mag * mask2
        est2_est_ref_pha = self.phase_decoder2(x2)
        
        est2_mag = est2_mix_est_mag + est2_est_ref_mag
        est2_pha = est2_mix_est_pha + est2_est_ref_pha
        
        return est2_mag, est2_pha, z
        
        
class SpeakerExtractor(nn.Module):
    
    def __init__(self, **kwargs):
        super(SpeakerExtractor, self).__init__()
        
        self.dim = kwargs['H'] // 2
        #self.n_fft = 256
        
        self.encoder = nn.Sequential(
            DenseEncoder(in_channel = 4, dim = self.dim),
            nn.GroupNorm(1, self.dim)
        )
        
        self.speaker_extractor = ConvNeXtV2(emb_dim = self.dim)
        #self.speaker_extractor = UNet(emb_dim = self.dim)

    def transform_mag_pha_half_frame(self, x):
    
        b, c, f, t = x.shape
        
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1)) # zero-padding
            t = t+1 # now, even
            
        x =  x.view(b, c, f, 2, t//2)
        x =  x.permute(0, 2, 1, 3, 4).reshape(b, 2*c, f, t//2)
        x =  x.transpose(3, 2)
        
        return x
    
    def inverse_transform_mag_pha_half_frame(self, x, original_t):
        
        b, ch2, t2, f = x.shape
        
        c = ch2 // 2
        
        x = x.transpose(3, 2)                 # [B, 2C, F, T/2]
        x = x.view(b, c, 2, f, t2)            # [B, C, 2, F, T/2]
        x = x.permute(0, 1, 3, 2, 4)          # [B, C, F, 2, T/2]
        x = x.reshape(b, c, f, 2 * t2)        # [B, C, F, even_T]
        x = x[..., :original_t]               # [B, C, F, original_t]
        
        return x
        
    def forward(self, mag_pha):
    
        _, _, _, original_t = mag_pha.shape
        
        x = self.transform_mag_pha_half_frame(mag_pha)
        x = self.encoder(x)
        z = self.speaker_extractor(x)
        
        return z
        
class CROCS(nn.Module):
    def __init__(self, device, **kwargs):
        super(CROCS, self).__init__()
        
        self.device = device
        
        self.tse_coarse  = CROCS_Coarse(device, **kwargs)
        self.tse_sv      = SpeakerExtractor(**kwargs)
        self.tse_fine    = CROCS_Fine(device, **kwargs)
        
        
        self.compression_factor = kwargs['compression_factor']
        
        self.n_fft          = kwargs['n_fft']
        self.window_length  = kwargs['window_length']
        self.hop_size       = kwargs['hop_size']
        
    def forward(self, mixture, reference):
        
        est1, mix_mag, mix_pha, est_mag, est_pha, ref_mag_pha = self.tse_coarse(mixture, reference)
        
        z_ref = self.tse_sv(ref_mag_pha)
        est_mag_pha = torch.stack([est_mag, est_pha], dim = 1)
        z_est = self.tse_sv(est_mag_pha)
        
        ref_mag = ref_mag_pha[:,0]
        ref_pha = ref_mag_pha[:,1]
        
        est2_mag, est2_pha, z = self.tse_fine(mix_mag, mix_pha, est_mag, est_pha, ref_mag, ref_pha, z_ref, z_est)
        
        length = mixture.shape[-1]
        
        est2 = _inverse_power_compression_and_istft(est2_mag, est2_pha, length, self.compression_factor, self.device, n_fft=self.n_fft, hop_size=self.hop_size, win_length=self.window_length)
        
        #z = torch.concat([z_ref, z_est], dim=-1)
        
        return est1, est2, z
    
    
if __name__ == '__main__':
    
    compression_factor = 0.3
    win_length = 256 # 32ms
    hop_size = 128
    n_fft   = 256
    
    if torch.cuda.is_available():
        device = torch.device("cuda");print(device)

    configuration = {
        'R_coarse' : 3,
        'R_fine' : 1,
        'M' : 4,
        'H' : 80,
        'n_fft': 256,
        'compression_factor': compression_factor,
        'fusion_type'        : 'PSM',
        'weight_spk'         : 0.5,
        'temperature'        : 4,
    }
    
    # sr = 8000,  L_mixture = 4.0s
    mixture     = torch.randn([1,1,32000]).to(device)
    enrollment  = torch.randn([1,1,32000]).to(device)
    
    #enrollment, sr = torchaudio.load('./enroll.wav')
    #mixture, _ = torchaudio.load('./mixture.wav')
    
    enrollment = enrollment[...,:32000]
    mixture = mixture[...,:32000]
    
    # mix_data = mixture.squeeze().cpu().detach().numpy()
    

    model = CROCS(device, **configuration).to(device)

    k = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
    print('# of parameters: {:.2f} M'.format(k))
    
    k2 = sum(p.numel() for p in model.tse_sv.speaker_extractor.parameters() if p.requires_grad) / 10**6
    print('# of parameters: {:.2f} M'.format(k2))
   
    model.eval()
    est1, est2, z = model(mixture, enrollment)
   
    print('\n------Result--------')
    print('estimated signals 1', est1.shape)
    print('estimated signals 2', est2.shape)
    # print('ref mag: {}, estimated ref mag: {}'.format(ref_mag.shape, est_ref_mag.shape))
    
    
    # def prepare_input(input_shape):
        
    #     inputs = {
    #         'mixture': torch.ones((1, *input_shape)).to(device),
    #         'reference': torch.ones((1, *input_shape)).to(device),
    #     }
        
    #     return inputs
    
    # model = CROCS_Coarse(device, **configuration).to(device)
    
    # # 2-seconds
    # macs1, params1 = get_model_complexity_info(model, (1, 32000), as_strings=True, backend='pytorch',
    #                                   print_per_layer_stat=False, verbose=True, input_constructor = prepare_input)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs1))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params1))
    
    # def prepare_input2(input_shape):
        
        
    #     inputs = {
    #         'mix_mag': torch.ones((1, *input_shape)).to(device),
    #         'mix_pha': torch.ones((1, *input_shape)).to(device),
    #         'est_mag': torch.ones((1, *input_shape)).to(device),
    #         'est_pha': torch.ones((1, *input_shape)).to(device),
    #         'ref_mag': torch.ones((1, *input_shape)).to(device),
    #         'ref_pha': torch.ones((1, *input_shape)).to(device),
    #         'z_ref': torch.ones((1, 40)).to(device),
    #         'z_est': torch.ones((1, 40)).to(device),
    
    #     }
        
    #     return inputs
    
    
    # model = CROCS_Fine(device, **configuration).to(device)
    
    # # 1-seconds 63
    # # 4-second 251
    
    # macs2, params2 = get_model_complexity_info(model, (129, 251), as_strings=True, backend='pytorch',
    #                                   print_per_layer_stat=False, verbose=True, input_constructor = prepare_input2)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs2))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params2))
    
    
    # model = SpeakerExtractor(**configuration).to(device)
    
    # def prepare_input3(input_shape):
        
    #     inputs = {
    #         'mag_pha': torch.ones((1, *input_shape)).to(device),
    #     }
    #     return inputs
    
    # macs3, params3 = get_model_complexity_info(model,  (2, 129, 251), as_strings=True, backend='pytorch',
    #                                   print_per_layer_stat=False, verbose=True, input_constructor = prepare_input3)
    
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs3))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params3))
    
    # def parse_ptflops(s: str) -> float:
        
    #     s = s.replace(',', '').strip()
    #     parts = s.split()
    #     num = float(parts[0])
    #     unit = parts[1].lower() if len(parts) > 1 else ''
    
    #     if unit.startswith('g'): mul = 1e9
    #     elif unit.startswith('m'): mul = 1e6
    #     elif unit.startswith('k'): mul = 1e3
    #     else: mul = 1.0
        
    #     return num * mul
    
    # def humanize(n, unit="", base=1000):
    #     for u in ["", "K", "M", "G", "T"]:
    #         if abs(n) < base:
    #             return f"{n:.2f} {u}{unit}".strip()
    #         n /= base
    #     return f"{n:.2f} P{unit}".strip()
    
    # macs_total = parse_ptflops(macs1) + parse_ptflops(macs2) + parse_ptflops(macs3)
    # params_total = parse_ptflops(params1) + parse_ptflops(params2) + parse_ptflops(params3)
    
    # print('-------------------------------------------------------------')
    # print("Computational complexity:", humanize(macs_total, "Mac"))  # 138.05 GMac
    # print("Number of parameters:", humanize(params_total))           # 6.96 M
        
    
    