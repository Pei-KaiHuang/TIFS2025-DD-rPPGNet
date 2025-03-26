import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft

from loss import NegPearsonLoss



def compare_samples_neg_pearson(list_a, list_b, exclude_same=False):
    # Ensure tensors are not modified in-place by cloning
    list_a = torch.stack(list_a)  # Shape: (N, D)
    list_b = torch.stack(list_b)  # Shape: (N, D)
    # Compute means
    mean_a = list_a.mean(dim=1, keepdim=True)
    mean_b = list_b.mean(dim=1, keepdim=True)

    # Normalize (zero mean)
    list_a = list_a - mean_a
    list_b = list_b - mean_b

    # Compute standard deviations
    std_a = list_a.norm(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
    std_b = list_b.norm(dim=1, keepdim=True) + 1e-8

    # Compute cosine similarity, which is equivalent to Pearson correlation for zero-mean data
    correlation_matrix = (list_a @ list_b.T) / (std_a * std_b.T)  # Shape: (N, N)

    # Convert Pearson correlation to loss
    pairwise_loss = 1 - correlation_matrix  # Negate to match NegPearsonLoss

    if exclude_same:
        # Mask diagonal elements (self-comparison)
        mask = 1 - torch.eye(pairwise_loss.size(0), device=pairwise_loss.device)
        pairwise_loss = pairwise_loss * mask

    return pairwise_loss.mean()


    
def compare_samples_cdist(list_a, list_b, exclude_same=False):
    # Ensure tensors are not modified in-place by cloning
    list_a = torch.stack(list_a)
    list_b = torch.stack(list_b)

    # Compute pairwise distances using cdist
    pairwise_dist = torch.cdist(list_a, list_b, p=2)

    if exclude_same:
        # Create a mask to exclude self-comparisons
        mask = 1 - torch.eye(pairwise_dist.size(0), device=pairwise_dist.device)
        pairwise_dist = pairwise_dist * mask

    return pairwise_dist.mean()

    
    

class ContrastLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass, PSD_or_signal='PSD'):
        super(ContrastLoss, self).__init__()
        
        
        if PSD_or_signal == 'PSD':
            self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
            self.distance_func = compare_samples_cdist
        elif PSD_or_signal == 'signal':
            self.ST_sampling = ST_sampling_signal(delta_t, K) # spatiotemporal sampler
            self.distance_func = compare_samples_neg_pearson
        else:
            raise ValueError('PSD_or_signal must be PSD or signal')

    
    def forward(self, X_rPPG):
        
        batch_size = X_rPPG.size(0)
        samples_X_rPPG = self.ST_sampling(X_rPPG)

        pos_loss_X_rPPG = 0
        count = 0
        for i in range(0, batch_size):
            pos_loss_X_rPPG += self.distance_func(samples_X_rPPG[i], samples_X_rPPG[i], exclude_same=True)
            count += 1
        pos_loss_X_rPPG = pos_loss_X_rPPG / count

        neg_loss_X_rPPG = 0
        count = 0
        for i in range(0, batch_size):
            for j in range(i + 1, batch_size):
                # print(i,j)
                neg_loss_X_rPPG += -self.distance_func(samples_X_rPPG[i], samples_X_rPPG[j])
                count += 1
        neg_loss_X_rPPG = neg_loss_X_rPPG / count


        numerator = torch.exp(pos_loss_X_rPPG)
        denominator = torch.exp(pos_loss_X_rPPG) + torch.exp(neg_loss_X_rPPG)
        total_loss = -torch.log(numerator / (denominator + 1e-8))
        return total_loss
    
    
    
    # for k-means
    def forward_k_means(self, anc, neg):
        
        samples_X_rPPG = self.ST_sampling(anc)
        samples_Y_rPPG = self.ST_sampling(neg)
        
        
        pos_loss_X_rPPG = 0
        count = 0
        
        
        anc_centroid = torch.mean(anc, dim=0).unsqueeze(0)
        samples_anc = self.ST_sampling(anc_centroid)

        
        for i in range(0, anc.shape[0]):
            pos_loss_X_rPPG += self.distance_func(samples_X_rPPG[i], samples_anc[0])
            count += 1
        pos_loss_X_rPPG = pos_loss_X_rPPG / count
            
        
        neg_loss_X_rPPG = 0
        count = 0
        for i in range(0, neg.shape[0]):
            neg_loss_X_rPPG -= self.distance_func(samples_Y_rPPG[i], samples_anc[0])
            count += 1
        neg_loss_X_rPPG = neg_loss_X_rPPG / count
        
        
        numerator = torch.exp(pos_loss_X_rPPG)
        denominator = torch.exp(pos_loss_X_rPPG) + torch.exp(neg_loss_X_rPPG)
        total_loss = -torch.log(numerator / (denominator + 1e-8)) 

        return total_loss
    
    
    def forward_pos_and_neg(self, pos, neg):
        
        samples_X_rPPG = self.ST_sampling(pos)
        samples_Y_rPPG = self.ST_sampling(neg)
        
        
        pos_loss_X_rPPG = 0
        count = 0
        for i in range(0, pos.shape[0]):
            pos_loss_X_rPPG += self.distance_func(samples_X_rPPG[i], samples_X_rPPG[i], exclude_same=True)
            count += 1
        pos_loss_X_rPPG = pos_loss_X_rPPG / count
        
        
        neg_loss_X_rPPG = 0
        count = 0
        for i in range(0, pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                neg_loss_X_rPPG += -self.distance_func(samples_X_rPPG[i], samples_X_rPPG[j])
                count += 1
        neg_loss_X_rPPG = neg_loss_X_rPPG / count
            
        
        neg_loss_Y_rPPG = 0
        count = 0
        for i in range(0, pos.shape[0]):
            for j in range(0, neg.shape[0]):
                neg_loss_Y_rPPG += -self.distance_func(samples_X_rPPG[i], samples_Y_rPPG[j])
                count += 1
        neg_loss_Y_rPPG = neg_loss_Y_rPPG / count
        
        
        numerator = torch.exp(pos_loss_X_rPPG)
        denominator = torch.exp(pos_loss_X_rPPG) + torch.exp(neg_loss_X_rPPG) + torch.exp(neg_loss_Y_rPPG)
        total_loss = -torch.log(numerator / (denominator + 1e-8))
        return total_loss
    
    
    def forward_anc_pos_neg(self, anc, pos, neg):
        
        samples_anc_rPPG = self.ST_sampling(anc)
        samples_pos_rPPG = self.ST_sampling(pos)
        samples_neg_rPPG = self.ST_sampling(neg)
        
        pos_loss_X_rPPG = 0
        count = 0
        for i in range(0, pos.shape[0]):
            pos_loss_X_rPPG += self.distance_func(samples_anc_rPPG[i], samples_pos_rPPG[i])
            count += 1
        
        pos_loss_X_rPPG = pos_loss_X_rPPG / count
        
        neg_loss_X_rPPG = 0
        count = 0
        for i in range(0, pos.shape[0]):
            neg_loss_X_rPPG += -self.distance_func(samples_anc_rPPG[i], samples_neg_rPPG[i])
            count += 1
        neg_loss_X_rPPG = neg_loss_X_rPPG / count
        
        numerator = torch.exp(pos_loss_X_rPPG)
        denominator = torch.exp(pos_loss_X_rPPG) + torch.exp(neg_loss_X_rPPG)
        total_loss = -torch.log(numerator / (denominator + 1e-8))
        return total_loss
    
    



class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input, fps_list=None): # input: (2, N, T)
        samples = []
        min_fps = None if fps_list is None else min(fps_list)
        
        for b in range(input.shape[0]): # loop over videos (totally 2 videos)
            cur_fps = None if fps_list is None else fps_list[b]
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    zero_pad = 0 if min_fps is None else (cur_fps/min_fps)-1.0
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t], 
                                      zero_pad=zero_pad,
                                      cur_fps=cur_fps)
                    
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples
    
    
    

class ST_sampling_signal(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, K):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position

    def forward(self, input, fps_list=None): # input: (2, N, T)
        samples = []
        min_fps = None if fps_list is None else min(fps_list)
        
        for b in range(input.shape[0]): # loop over videos (totally 2 videos)
            cur_fps = None if fps_list is None else fps_list[b]
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    # zero_pad = 0 if min_fps is None else (cur_fps/min_fps)-1.0
                    # x = self.norm_psd(input[b, c, offset:offset + self.delta_t], 
                    #                   zero_pad=zero_pad,
                    #                   cur_fps=cur_fps)
                    
                    x = input[b, c, offset:offset + self.delta_t]
                    
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples




class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0, cur_fps=None):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        if cur_fps is not None:
            Fn = cur_fps / 2
        else:
            Fn = self.Fs / 2
            
            
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        
        # print(zero_pad, cur_fps, Fn, x.shape)
        
        return x


if __name__ == '__main__':
    # test the contrastive loss
    loss_func = ContrastLoss(delta_t=150, K=4, Fs=30, high_pass=40, low_pass=180)
    anc = torch.randn(5, 5, 300).to('cuda')
    anc = torch.randn(5, 5, 300).to('cuda')
    
    start_time = time.time()
    
    for i in range(100):
        loss = loss_func(anc, neg)
    
    print(f"Time: {time.time()-start_time}")
    
    
    