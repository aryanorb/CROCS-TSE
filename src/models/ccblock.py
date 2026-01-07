import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.nn.functional as F


class CrossCorrelationBlock(nn.Module):
    def __init__(self, fusion_type, T=4.0, eps=1e-6):
        super(CrossCorrelationBlock, self).__init__()
        self.fusion_type = fusion_type
        self.T = T
        self.eps = eps

    def forward(self, mix_magnitude, mix_phase, ref_magnitude, ref_phase):
        # [B, F, T_mix]
        
        #print(mix_magnitude.shape, mix_phase.shape, ref_magnitude.shape, ref_phase.shape)
        
        mix_real = mix_magnitude * torch.cos(mix_phase)
        mix_imag = mix_magnitude * torch.sin(mix_phase)

        # [B, F, T_ref]
        ref_real = ref_magnitude * torch.cos(ref_phase)
        ref_imag = ref_magnitude * torch.sin(ref_phase)

        # [B, T_mix ,F]
        mix_real_T = mix_real.transpose(1, 2)
        mix_imag_T = mix_imag.transpose(1, 2)

        # [B, T_ref ,F]
        ref_real_T = ref_real.transpose(1, 2)
        ref_imag_T = ref_imag.transpose(1, 2)

        # Cross-correlation components: [B, T_mix, T_ref]
        c_real = torch.matmul(mix_real_T, ref_real) + torch.matmul(mix_imag_T, ref_imag)
        c_imag = torch.matmul(mix_imag_T, ref_real) - torch.matmul(mix_real_T, ref_imag)

        # ---------- CIENet ----------
        if self.fusion_type == "CIENet":
            # real branch
            s_real = torch.matmul(mix_real_T, ref_real)          # [B, Tm, Tr]
            a_real = F.softmax(s_real, dim=-1)
            fusion_real_T = torch.matmul(a_real, ref_real_T)     # [B, Tm, F]

            # imag branch
            s_imag = torch.matmul(mix_imag_T, ref_imag)          # [B, Tm, Tr]
            a_imag = F.softmax(s_imag, dim=-1)
            fusion_imag_T = torch.matmul(a_imag, ref_imag_T)     # [B, Tm, F]

            # [B, Tm, F] -> [B, F, Tm]
            z_real = fusion_real_T.transpose(1, 2)
            z_imag = fusion_imag_T.transpose(1, 2)
            

        elif self.fusion_type == "PSM":
            logits = c_real / self.T
            w = F.softmax(logits, dim=-1)
            z_real_T = torch.matmul(w, ref_real_T)
            z_imag_T = torch.matmul(w, ref_imag_T)
            
            z_real = z_real_T.transpose(1, 2)
            z_imag = z_imag_T.transpose(1, 2)
            

        elif self.fusion_type == "IRM":
            c_mag = torch.sqrt(c_real**2 + c_imag**2 + self.eps)
            logits = c_mag / self.T
            w = F.softmax(logits, dim=-1)
            z_real_T = torch.matmul(w, ref_real_T)
            z_imag_T = torch.matmul(w, ref_imag_T)
            
            z_real = z_real_T.transpose(1, 2)
            z_imag = z_imag_T.transpose(1, 2)

        else:
            raise RuntimeError("check your fusion type: ['PSM','IRM','CIENet']")
        
        mag = torch.sqrt(z_real**2 + z_imag**2 + self.eps)
        pha = torch.atan2(z_imag, z_real)
        
        return mag, pha