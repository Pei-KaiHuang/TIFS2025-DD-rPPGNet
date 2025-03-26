from PhysNetModel import PhysNet
import argparse

from dataloader import get_loader
from loss import *


from util import *
from dataloader import get_loader

from einops import rearrange


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

else:
    device = torch.device('cpu')


args = get_args()
trainName, testName = get_name(args)
log = get_logger(f"logger/test/{args.test_dataset}/", testName)

print(f"{trainName=}")
print(f"{testName=}")

# TODO: test_T represents the length of each video. If the length of each video in your dataset is a different value, you should modify it accordingly.
seq_len = args.test_T*args.fps
test_loader = get_loader(_datasets=list(args.test_dataset), 
                        _seq_length=seq_len,
                        batch_size=args.bs,
                        shuffle=False,
                        train=False,
                        if_bg=False,)

# TODO: Change the model path
model_path = f"/shared/DD-rPPGNet/results/{args.train_dataset}/{trainName}/weight"
model = PhysNet(S=args.model_S,
                in_ch=args.in_ch, 
                conv_type=args.conv).to(device).train()



with torch.no_grad():

    for epoch in range(0, args.epoch):
        model.load_state_dict(torch.load(model_path + f'/fg_epoch{epoch}.pt', map_location=device))
        

        # Specify the functions and initialize metric lists
        functions = {
            "HR": (predict_heart_rate_batch, [], [], []),
        }
        
        for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):
            datalength = face_frames.size(2)
            num_batch_per_sample = datalength //seq_len
            # (B, C, datalength, 64, 64) to (B*num_batch_per_sample, 3, seq_len, 64, 64)
            
            imgs = rearrange(face_frames, 'b c (t1 t2) h w -> (b t1) c t2 h w', t1=num_batch_per_sample).to(device)
            _label = rearrange(ppg_label, 'b (t1 t2) -> (b t1) t2', t1=num_batch_per_sample).detach().cpu().numpy()
            #print(f"{imgs.shape=}, {_label.shape=}")
            rPPG = model(imgs)[:, -1]

            rppg = rPPG.detach().cpu().numpy()
            rppg = butter_bandpass_batch(rppg, lowcut=0.6, highcut=4, fs=args.fps)
            _label = butter_bandpass_batch(_label, lowcut=0.6, highcut=4, fs=args.fps)

            # Processing with the specified functions
            for func_name, (func, mae_list, rmse_list, r_list) in functions.items():
                preds = func(rppg.copy(), fs=args.fps)
                lbls = func(_label.copy(), fs=args.fps)
            
                for i in range(0, len(preds), num_batch_per_sample):
                    pred = preds[i:i+num_batch_per_sample]
                    lbl = lbls[i:i+num_batch_per_sample]

                    mae = np.mean(np.abs(pred - lbl))
                    rmse = np.sqrt(np.mean((pred - lbl) ** 2))
                    pearson_corr = Pearson_np(pred, lbl)

                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    r_list.append(pearson_corr)

                    log.info(f'[epoch {epoch} step {step} mae {mae:>8.5f} rmse {rmse:>8.5f} pearson_corr {pearson_corr:>8.5f}] {subjects[i//num_batch_per_sample]:<12} func {func_name}')


        # Logging the average metrics
        for func_name, (_, mae_list, rmse_list, r_list) in functions.items():
            log.info(f'[epoch {epoch} avg all_mae {np.mean(mae_list):>8.5f} all_rmse {np.mean(rmse_list):>8.5f} all_R {np.mean(r_list):>8.5f}] (func {func_name})')
