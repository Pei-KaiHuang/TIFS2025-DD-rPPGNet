import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyContrast(nn.Module):
    """ Frequency contrast wrapper around a backbone model e.g. PhysNet
    """
    def __init__(self, model, device):
        super().__init__()

        # self.backbone = init_model(args.contrast_model, args, device, dataset)
        self.backbone = model
        self.upsampler = nn.Upsample(size=(300), mode='linear', align_corners=False)

    def forward(self, x_a, noise_a=None):
        B = x_a.shape[0]
        D = x_a.shape[2]
        branches = {}

        # Resample input
        freq_factor = torch.FloatTensor(1).uniform_(0.8, 1.4).to(x_a.device)
        target_size = int(D / freq_factor)
        resampler = nn.Upsample(size=(target_size, x_a.shape[3], x_a.shape[4]),
                                mode='trilinear',
                                align_corners=False)
        x_n = resampler(x_a)
        if x_n.shape[2] < D:
            x_n = F.pad(x_n, (0, 0, 0, 0, 0, D - target_size))
        else:
            x_n = x_n[:, :, :D]


        if noise_a is None:
            multi_spatial_y = self.backbone(x_a)
        else:
            multi_spatial_y, multi_spatial_noise_y, multi_spatial_noisy_rPPG_y, fg_to_bg_noise = self.backbone(x_a, noise_a)

        y_a = multi_spatial_y[:, -1].unsqueeze(1)
        y_n = self.backbone(x_n)[:, -1].unsqueeze(1)

        # Remove padding from negative branch
        y_n = y_n[:,:,:target_size]

        # Resample negative PPG to create positive branch
        y_p = self.upsampler(y_n)

        return y_a, y_p, y_n




if __name__ == "__main__":
    
    from PhysNetModel import PhysNet
    
    model = PhysNet(S=2, in_ch=3).to('cuda').train()
    
    x = torch.rand(5, 3, 300, 64, 64).to('cuda')
    
    freq_contrast = FrequencyContrast(model, 'cuda')
    
    y_a, y_p, y_n = freq_contrast(x)
    print(y_a.shape, y_p.shape, y_n.shape)
    
    
