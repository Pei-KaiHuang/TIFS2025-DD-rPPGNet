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

        
        functions = {
            "HR": predict_heart_rate_batch,
        }

        all_preds = {fn: [] for fn in functions}
        all_lbls = {fn: [] for fn in functions}

        for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):
            datalength = face_frames.size(2)
            num_batch_per_sample = datalength // seq_len

            imgs = rearrange(face_frames, 'b c (t1 t2) h w -> (b t1) c t2 h w', t1=num_batch_per_sample).to(device)
            _label = rearrange(ppg_label, 'b (t1 t2) -> (b t1) t2', t1=num_batch_per_sample).detach().cpu().numpy()

            rPPG = model(imgs)[:, -1].detach().cpu().numpy()

            # bandpass filter
            rppg = butter_bandpass_batch(rPPG, lowcut=0.6, highcut=4, fs=args.fps)
            _label = butter_bandpass_batch(_label, lowcut=0.6, highcut=4, fs=args.fps)

            for func_name, func in functions.items():
                preds = func(rppg.copy(), fs=args.fps)
                lbls = func(_label.copy(), fs=args.fps)

                all_preds[func_name].extend(preds)
                all_lbls[func_name].extend(lbls)

        
        # Calculate the metrics
        for func_name in functions:
            preds = np.array(all_preds[func_name])
            lbls = np.array(all_lbls[func_name])
            print(f"preds: {preds.shape}, lbls: {lbls.shape}")

            mae = np.mean(np.abs(preds - lbls))
            rmse = np.sqrt(np.mean((preds - lbls) ** 2))
            pearson_corr = Pearson_np(preds, lbls)

            log.info(f'[epoch {epoch}] METRIC for {func_name}: MAE={mae:>8.5f}, RMSE={rmse:>8.5f}, R={pearson_corr:>8.5f}')
