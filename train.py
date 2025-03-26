import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim

from einops import rearrange

from PhysNetModel import PhysNet
from loss import NegPearsonLoss, ContrastLoss, FrequencyContrast
from util import *
from dataloader import get_loader


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    


args = get_args()
trainName, _ = get_name(args)
log = get_logger(f"logger/train/{args.train_dataset}/", trainName)

print(f"{trainName=}")


# TODO: modify the path to store the training model
result_dir = f"/shared/DD-rPPGNet/results/{args.train_dataset}/{trainName}/"
os.makedirs(f"{result_dir}/weight", exist_ok=True)

seq_len = args.train_T*args.fps
train_loader = get_loader(_datasets=list(args.train_dataset),
                          _seq_length=seq_len,
                          batch_size=args.bs,
                          train=True,
                          if_bg=True,)

model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv).to(device).train()

# define irrelevant power ratio
IPR = IrrelevantPowerRatio(Fs=args.fps, 
                           high_pass=args.high_pass, low_pass=args.low_pass)

# define loss
loss_np = NegPearsonLoss()

loss_func_rPPG = ContrastLoss(delta_t=seq_len//2, K=4, Fs=args.fps, 
                              high_pass=args.high_pass, low_pass=args.low_pass,
                              PSD_or_signal='PSD')
loss_func_rPPG_Sig = ContrastLoss(delta_t=seq_len//2, K=4, Fs=args.fps, 
                              high_pass=args.high_pass, low_pass=args.low_pass,
                              PSD_or_signal='signal')

freq_contrast = FrequencyContrast(model, device)

opt_fg = optim.AdamW(model.parameters(), lr=args.lr)


for epoch in range(args.epoch):
    
    print(f"epoch_train: {epoch}/{args.epoch}:")
    for step, (face_frames, face_frames_aug, bg_frames, _, subjects) in enumerate(train_loader):
        
        # if batch size < 2, skip this round since we need at least 2 samples for the contrastive loss
        if(face_frames.shape[0] < 2):
            continue
        
        face_frames = face_frames.to(device)
        face_frames_aug = face_frames_aug.to(device)
        bg_frames = bg_frames.to(device)
        
        
        multi_spatial_y, multi_spatial_noise_y, multi_spatial_noisy_rPPG_y, fg_to_bg_noise = model(face_frames, bg_frames)

        multi_spatial_y_aug = model(face_frames_aug, None, if_de_interfered=True)
        multi_spatial_noisy_y_aug = model(face_frames_aug, None, if_de_interfered=False)
        
        

        
        # Eq (5)
        loss_noise_np = loss_np(multi_spatial_noise_y[:, -1], fg_to_bg_noise[:, -1])

        # Eq (6)
        all_noises = torch.cat([multi_spatial_noise_y, fg_to_bg_noise], dim=0)
        all_noises = rearrange(all_noises, 'b s d -> (b s) d')
        clusters = {}
        n_clusters=4
        noise_labels, noise_centroids = KMeans(all_noises, n_clusters=n_clusters)
        for i in range(n_clusters):
            clusters[i] = all_noises[noise_labels == i]
        
        loss_kmeans = 0
        for i in range(n_clusters):
            anc = clusters[i]
            if anc.shape[0] < 2:
                continue
            
            neg = torch.cat([clusters[j] for j in range(n_clusters) if j != i], dim=0)
            
            anc = anc.unsqueeze(1)
            neg = neg.unsqueeze(1)
            # print(f"{anc.shape=}, {neg.shape=}")
            
            if anc.shape[0] < 2 or neg.shape[0] < 2:
                continue
            
            loss_tmp = loss_func_rPPG_Sig.forward_k_means(anc, neg)
            loss_kmeans += loss_tmp
            
        # Eq (7)
        all_noisy_ppg = torch.cat([multi_spatial_noisy_rPPG_y, multi_spatial_noisy_y_aug], dim=1)
        loss_contrastive_cr = loss_func_rPPG(all_noisy_ppg)


        # Eq (8)
        all_ppg = torch.cat([multi_spatial_y, multi_spatial_y_aug], dim=1)
        loss_contrastive_dcr = loss_func_rPPG.forward_pos_and_neg(all_ppg, fg_to_bg_noise)


        # Temporal augmentation
        y_a, y_p, y_n = freq_contrast(face_frames, bg_frames)
        loss_freq_contrast = loss_func_rPPG.forward_anc_pos_neg(y_a, y_p, y_n)
        
        opt_fg.zero_grad()
        total_loss = loss_noise_np*0.05 + loss_kmeans*0.05 + loss_contrastive_cr*0.5 + loss_contrastive_dcr + loss_freq_contrast*0.5 
        total_loss.backward()
        opt_fg.step()
        
        
        # evaluate irrelevant power ratio during training
        ipr = torch.mean(IPR(multi_spatial_y[:, -1].clone().detach()))
        
        
        loss_string =  f"[epoch {epoch} step {step}]"
        loss_string += f" loss_noise_np: {loss_noise_np.item():.4f}"
        if loss_kmeans != 0:
            loss_string += f" loss_kmeans: {loss_kmeans.item():.4f}"
        loss_string += f" loss_contrastive_cr: {loss_contrastive_cr.item():.4f}"
        loss_string += f" loss_contrastive_dcr: {loss_contrastive_dcr.item():.4f}"
        loss_string += f" loss_freq_contrast: {loss_freq_contrast.item():.4f}"

        loss_string += f" IPR: {ipr.item():.4f}"
        log.info(loss_string)
        # exit()
        
                
        
        
        
    torch.save(model.state_dict(), result_dir + '/weight/fg_epoch%d.pt' % epoch) 


