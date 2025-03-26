import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



"""
Normal LDC
"""
class LDC_M(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        
        super(LDC_M, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
                
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.05, requires_grad=True)
        
        [_, _, t, h, w] = self.conv.weight.shape
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=True)
        # self.learnable_mask = nn.Parameter(torch.ones([t, h, w]) * 0.1, requires_grad=True)
        
        
    def forward(self, x):
        
        # [_in, _out, _, _, _] = self.conv.weight.shape
        # ldp_mask = self.learnable_mask.expand(_in, _out, -1, -1, -1)
        mask = (1-self.learnable_theta)*self.base_mask + self.learnable_theta*self.learnable_mask
        
        out_diff = F.conv3d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        
        return out_diff

# -------------------------------------------------------------------------------------------------------------------
# PhysNet model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
class _PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv3x3x3=nn.Conv3d):
        super().__init__()

        self.S = S  # S is the spatial dimension of ST-rPPG block

        self.encoder1_entangle = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder2_entangle = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # rPPG 
        self.encoder3_rPPG = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # rPPG 
        self.encoder3_noise = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1_rPPG = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2_rPPG = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.end_rPPG = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

        # noise 
        self.decoder1_nosie = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2_noise = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.end_noise = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )
 

    def forward(self, x, y=None, if_de_interfered=True):
        # x is fg, y is bg 
        if y is not None:

            means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
            stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
            x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

            # nosie image, y 
            means_y = torch.mean(y, dim=(2, 3, 4), keepdim=True)
            stds_y = torch.std(y, dim=(2, 3, 4), keepdim=True)
            y = (y - means_y) / stds_y  # (B, C, T, 128, 128)

            parity_y = []
            y = self.encoder1_entangle(y)  # (B, C, T, 128, 128) 
            parity_y.append(y.size(2) % 2)
            y_entangle = self.encoder2_entangle(y)  # (B, 64, T/2, 32, 32)
            parity_y.append(y_entangle.size(2) % 2)
            y_noise = self.encoder3_noise(y_entangle)  # (B, 64, T/4, 16, 16)

            y = F.interpolate(y_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            y = self.decoder1_nosie(y)  # (B, 64, T/2, 8, 8)
            y = F.pad(y, (0, 0, 0, 0, 0, parity_y[-1]), mode='replicate')
            y = F.interpolate(y, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            y = self.decoder2_noise(y)  # (B, 64, T, 8, 8)
            y = F.pad(y, (0, 0, 0, 0, 0, parity_y[-2]), mode='replicate')
            y = self.end_noise(y)  # (B, 1, T, S, S), ST-rPPG block

            y_list = []
            for a in range(self.S):
                for b in range(self.S):
                    y_list.append(y[:, :, :, a, b])  # (B, 1, T)

            y = sum(y_list) / (self.S * self.S)  # (B, 1, T)
            Y = torch.cat(y_list + [y], 1)  # (B, N, T), flatten all spatial signals to the second dimension

            # rPPG image
            parity_x = []
            x = self.encoder1_entangle(x)  # (B, C, T, 128, 128) 
            parity_x.append(x.size(2) % 2)
            x_mix = self.encoder2_entangle(x)  # (B, 64, T/2, 32, 32)
            parity_x.append(x_mix.size(2) % 2)
            x_rPPG = self.encoder3_rPPG(x_mix)  # (B, 64, T/4, 16, 16)
            x_noise = self.encoder3_noise(x_mix) 

            x = F.interpolate(x_rPPG - x_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            x = self.decoder1_rPPG(x)  # (B, 64, T/2, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-1]), mode='replicate')
            x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            x = self.decoder2_rPPG(x)  # (B, 64, T, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-2]), mode='replicate')
            x = self.end_rPPG(x)  # (B, 1, T, S, S), ST-rPPG block

            x_list = []
            for a in range(self.S):
                for b in range(self.S):
                    x_list.append(x[:, :, :, a, b])  # (B, 1, T)

            x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
            X = torch.cat(x_list + [x], 1)  # (B, N, T), flatten all spatial signals to the second dimension
            
            
            
            
            
            # TMM part
            # No cancellation rPPG
            
            z = F.interpolate(x_rPPG, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            z = self.decoder1_rPPG(z)  # (B, 64, T/2, 8, 8)
            z = F.pad(z, (0, 0, 0, 0, 0, parity_x[-1]), mode='replicate')
            z = F.interpolate(z, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            z = self.decoder2_rPPG(z)  # (B, 64, T, 8, 8)
            z = F.pad(z, (0, 0, 0, 0, 0, parity_x[-2]), mode='replicate')
            z = self.end_rPPG(z)  # (B, 1, T, S, S), ST-rPPG block

            z_list = []
            for a in range(self.S):
                for b in range(self.S):
                    z_list.append(z[:, :, :, a, b])  # (B, 1, T)

            z = sum(z_list) / (self.S * self.S)  # (B, 1, T)
            Z = torch.cat(z_list + [z], 1)  # (B, N, T), flatten all spatial signals to the second dimension
            
            
            # foreground forward to noise module
            xy = F.interpolate(x_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            xy = self.decoder1_nosie(xy)  # (B, 64, T/2, 8, 8)
            xy = F.pad(xy, (0, 0, 0, 0, 0, parity_y[-1]), mode='replicate')
            xy = F.interpolate(xy, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            xy = self.decoder2_noise(xy)  # (B, 64, T, 8, 8)
            xy = F.pad(xy, (0, 0, 0, 0, 0, parity_y[-2]), mode='replicate')
            xy = self.end_noise(xy)  # (B, 1, T, S, S), ST-rPPG block

            xy_list = []
            for a in range(self.S):
                for b in range(self.S):
                    xy_list.append(xy[:, :, :, a, b])  # (B, 1, T)

            xy = sum(xy_list) / (self.S * self.S)  # (B, 1, T)
            XY = torch.cat(xy_list + [xy], 1)  # (B, N, T), flatten all spatial signals to the second dimension
            
            
            
            return X, Y, Z, XY
        
        
        else:
            means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
            stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
            x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

            parity_x = []
            x = self.encoder1_entangle(x)  # (B, C, T, 128, 128) 
            parity_x.append(x.size(2) % 2)
            x_mix = self.encoder2_entangle(x)  # (B, 64, T/2, 32, 32)
            parity_x.append(x_mix.size(2) % 2)
            x_rPPG = self.encoder3_rPPG(x_mix)  # (B, 64, T/4, 16, 16)
            x_noise = self.encoder3_noise(x_mix)

            if if_de_interfered:
                x = F.interpolate(x_rPPG - x_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            else:
                x = F.interpolate(x_rPPG, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            x = self.decoder1_rPPG(x)  # (B, 64, T/2, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-1]), mode='replicate')
            x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            x = self.decoder2_rPPG(x)  # (B, 64, T, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-2]), mode='replicate')
            x = self.end_rPPG(x)  # (B, 1, T, S, S), ST-rPPG block

            x_list = []
            for a in range(self.S):
                for b in range(self.S):
                    x_list.append(x[:, :, :, a, b])  # (B, 1, T)

            x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
            X = torch.cat(x_list + [x], 1)
            return X
    
    
    def get_shallow_feature(self, x):
        means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

        parity_x = []
        x = self.encoder1_entangle(x)  # (B, C, T, 128, 128) 
        parity_x.append(x.size(2) % 2)
        x_mix = self.encoder2_entangle(x)  # (B, 64, T/2, 32, 32)

        return x_mix


class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv_type=None):
        super().__init__()
        
        # Our
        if conv_type == "LDC_M":
            print('Using LDC_M convolutions')
            conv3x3x3 = LDC_M
        else:
            print('Using vanilla 3D convolutions')
            conv3x3x3 = nn.Conv3d
        
        self.model = _PhysNet(S, in_ch, conv3x3x3)
        
    def forward(self, x, y=None, if_de_interfered=True):
        return self.model(x, y, if_de_interfered)
    
    
    
    def get_shallow_feature(self, x):
        return self.model.get_shallow_feature(x)
        
    

if __name__ == "__main__":
    
    H, W = 200, 200
    x = torch.randn([2, 3, 300, H, W]).cuda()
    y = torch.randn([2, 3, 300, H, W]).cuda()
    model = PhysNet(conv_type="LDC_M").cuda()
    # X, Y, Z, XY = model(x, y)
    
    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    
    
    # W = X + Y
    # print(W.shape)
    
    
    shallow = model.get_shallow_feature(x)
    print(shallow.shape)