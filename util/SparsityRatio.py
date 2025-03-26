import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft
from utils_sig import *




class SparsityRatio(nn.Module):
    # we reuse the code in Gideon2021 to get irrelevant power ratio
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass, delta=10):
        super(SparsityRatio, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.delta = delta

    def forward(self, preds):
        
        HR = predict_heart_rate(preds[0].cpu().clone().numpy(), Fs=self.Fs)
        
        # test_freqs, ps = compute_power_spectrum(preds[0].cpu().clone().numpy(), 30, zero_pad=100)
        # print(test_freqs.shape)
        # test_freqs2 = torch.fft.rfft(preds, dim=-1, norm='forward')
        # print(test_freqs2.shape)
        # Get PSD
        X_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm='forward'))

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X_real.shape[-2])
        all_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        peak_freqs = torch.logical_and((HR - self.delta) / 60 <= freqs, freqs <= (HR + self.delta) / 60)
        
        all_energy = tr.sum(tr.linalg.norm(X_real[:,all_freqs], dim=-1), dim=-1)
        peak_energy = tr.sum(tr.linalg.norm(X_real[:,peak_freqs], dim=-1), dim=-1)
        energy_ratio = tr.ones_like(all_energy)
        for ii in range(len(all_energy)):
            if all_energy[ii] > 0:
                energy_ratio[ii] = peak_energy[ii] / all_energy[ii]
                
        # print("In SparsityRatio, energy_ratio: ", energy_ratio)
        return energy_ratio




def sparsity_ratio_np(signal, Fs=30, high_pass=40., low_pass=250., delta=3.):
    
    # print(signal.shape)
    signal = signal - np.mean(signal)
    _freqs, _ps = compute_power_spectrum(signal, Fs, zero_pad=100)
    freqs_valid = np.logical_and(_freqs >= high_pass, _freqs <= low_pass)
    freqs = _freqs[freqs_valid]
    ps = _ps[freqs_valid]
    max_ind = np.argmax(ps)
    if 0 < max_ind < len(ps)-1:
        inds = [-1, 0, 1] + max_ind
        x = ps[inds]
        f = freqs[inds]
        d1 = x[1]-x[0]
        d2 = x[1]-x[2]
        offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
        if d2 > d1:
            offset *= -1
        max_bpm = f[1] + offset
    elif max_ind == 0:
        x0, x1 = ps[0], ps[1]
        f0, f1 = freqs[0], freqs[1]
        max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
    elif max_ind == len(ps) - 1:
        x0, x1 = ps[-2], ps[-1]
        f0, f1 = freqs[-2], freqs[-1]
        max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        
        
    peak_freqs = np.logical_and((max_bpm - delta) <= _freqs, _freqs <= (max_bpm + delta))
    peak_energy = _ps[peak_freqs]
    # print("In sparsity_ratio_np, ps: ", ps.shape)
    # print("In sparsity_ratio_np, peak_energy: ", peak_energy.shape)
    
    
    ratio = np.sum(peak_energy) / np.sum(ps)
    
    # print(ps)
    
    
    return max_bpm, ps, ratio
        
        

if __name__ == "__main__":
    
    a = np.random.randn(300)
    test_freqs, ps = compute_power_spectrum(a, 30, zero_pad=100)
    print(test_freqs)
    print(a.shape)
    print(test_freqs.shape)
    print(ps.shape)
    
    
    
    sparsity_ratio_np(a, Fs=30, high_pass=40., low_pass=250., delta=10.)