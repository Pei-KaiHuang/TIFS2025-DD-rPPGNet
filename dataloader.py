from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import glob
import random

from io import BytesIO
from tqdm import tqdm

START_OFFSET=30

from util import *
args = get_args()

def preload_frames(paths, transform_face, gray=False):
    """Loads and processes frames from given paths into memory."""
    frames = []

    for path in paths:
        frame = Image.open(path)
        if gray:
            frame = frame.convert('L')
        processed_frame = transform_face(frame)
        frames.append(processed_frame)

    return frames



def get_rPPG(path):

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    # hr = [float(ppg) for ppg in lines[1].split()[:100]]
    # no = [float(ppg) for ppg in lines[2].split()[:100]]
    f.close()

    return PPG


class rPPG_Dataset(Dataset):
    def __init__(self, datasets, seq_length, train, if_bg=True):
        
        self.train = train
        self.if_bg = if_bg
        self.gray = True if datasets == ["M"] else False

        # TODO: Change the path to adapt to your dataset
        face_folder = "crop_MTCNN"
        prefix = "/shared/rPPG_dataset"
        self.root_dir = {
            "P" : f"{prefix}/pure/{face_folder}",
            "U" : f"{prefix}/UBFC/{face_folder}",
            "C" : f"{prefix}/COHFACE/{face_folder}_30fps",
        }
        
        bg_folder = "bg_MTCNN"
        self.root_bg_dir = {
            "P" : f"{prefix}/pure/{bg_folder}",
            "U" : f"{prefix}/UBFC/{bg_folder}",
            "C" : f"{prefix}/COHFACE/{bg_folder}_30fps",
        }
        
        if train:
            self.subject_file = {
                "P" : f"./config/PURE_train.txt",
                "U" : f"./config/UBFC_train_split.txt",
                "C" : f"./config/COHFACE_train.txt",
            }
        else: # TODO: Modify the path to all when you run on the cross-dataset
            self.subject_file = {
                "P" : f"./config/PURE_test.txt",
                "U" : f"./config/UBFC_test_split.txt",
                "C" : f"./config/COHFACE_test.txt",
            }
        
        self.subjects= {}
        
        self.preloaded_subject_images = {}
        self.preloaded_bg_images = {}
        self.subject_GT_PPG = {}
        
        
        if self.train:
            _h, _w = 64, 64
        else:
            _h, _w = 64, 64

        self.transform_face = transforms.Compose([
            transforms.Resize((_h, _w)),
            transforms.ToTensor()
        ])


        # Adjust the initialization to preload images into memory
        for key in datasets:
            assert key in self.root_dir

            # Preload images and store them into the dictionary
            self.preload=True
            
            with open(self.subject_file[key]) as file:
                self.subjects[key] = [line.rstrip() for line in file]
                print(self.subjects[key])

            print("Loading :", key)
            for _subject in tqdm(self.subjects[key]):
                def get_files(path, subj):
                    files = []
                    for ext in ('*.png', '*.jpg'):
                        files.extend(sorted(glob.glob(os.path.join(path, subj, ext))))
                    return files
                
                image_paths = get_files(self.root_dir[key], _subject)
                bg_paths = get_files(self.root_bg_dir[key], _subject)
                
                _key = f"{key}_{_subject}"
                
                
                # Preload images and store them into the dictionary
                self.preload=True
                if self.preload:
                    # TODO: Modify the path to save the preprocessed data
                    prefix = "/shared/DD-rPPGNet/preprocessed_data"
                    if not os.path.exists(f'{prefix}/{_key}_fg.pt'):
                        self.preloaded_subject_images[_key] = preload_frames(image_paths, self.transform_face, gray=self.gray)
                        os.makedirs(prefix, exist_ok=True)
                        torch.save(self.preloaded_subject_images[_key], f'{prefix}/{_key}_fg.pt')

                    else:
                        self.preloaded_subject_images[_key] = torch.load(f'{prefix}/{_key}_fg.pt')
                
                else:
                    self.preloaded_subject_images[_key] = image_paths
                
                if self.if_bg:
                    
                    if self.preload:
                        if not os.path.exists(f'{prefix}/{_key}_bg.pt'):
                            self.preloaded_bg_images[_key] = preload_frames(bg_paths, self.transform_face, gray=self.gray)
                            os.makedirs(prefix, exist_ok=True)
                            torch.save(self.preloaded_bg_images[_key], f'{prefix}/{_key}_bg.pt')
                        else:
                            self.preloaded_bg_images[_key] = torch.load(f'{prefix}/{_key}_bg.pt')
                            
                    else:
                        self.preloaded_bg_images[_key] = bg_paths
                
                            
                    
                
                ground_truth = os.path.join(self.root_dir[key], _subject, "ground_truth.txt")
                self.subject_GT_PPG[_key] = get_rPPG(ground_truth)
        
        
        self.all_keys = list(self.preloaded_subject_images.keys())
        self.seq_length = seq_length
            

    def rotate_frames(self, frames, angle):

        """Rotates frames by a specified angle."""
        from torchvision.transforms.functional import rotate
        # Rotate each frame in the batch
        rotated_frames = torch.stack([rotate(frame, angle) for frame in frames])
        return rotated_frames


    def flip_frames(self, frames, horizontal=True):

        """Flips frames horizontally or vertically."""
        if horizontal:
            # Flip horizontally
            flipped_frames = torch.flip(frames, dims=[-1])  # Flip along the last dimension (width)
        else:
            # Flip vertically
            flipped_frames = torch.flip(frames, dims=[-2])  # Flip along the second-to-last dimension (height)
        return flipped_frames

            

    def __getitem__(self, idx):
        
        """Returns a dataset item given an index."""
        _key = self.all_keys[idx]
        _face_frame, _bg_frame, _ppg = [], [], []
        
        start = START_OFFSET if not self.train else random.randint(START_OFFSET, len(self.preloaded_subject_images[_key]) - self.seq_length)
        data_length = min(len(self.preloaded_subject_images[_key]), len(self.subject_GT_PPG[_key]))

        test_sample = data_length // self.seq_length

        num_batch = 1 if self.train else test_sample

        if self.preload:
            _face_frame_batch = self.preloaded_subject_images[_key][start:start+self.seq_length*num_batch]
        else:
            _face_frame_batch = preload_frames(self.preloaded_subject_images[_key][start:start+self.seq_length*num_batch], self.transform_face, gray=self.gray)
            
        _face_frame = torch.stack(_face_frame_batch).transpose(0, 1)
            
        # _face_frame_batch = self.preloaded_subject_images[_key][start:start+self.seq_length*num_batch]
        # _face_frame = torch.stack(_face_frame_batch).transpose(0, 1)
        
        # Apply transformations (rotation and flip)
        _face_frame_aug = []
        if self.train:  # Apply augmentations only during training

            # Random rotation (e.g., ±15 degrees)
            angle = random.uniform(-15, 15)
            _face_frame_aug = self.rotate_frames(_face_frame, angle)
            
            # Random horizontal flip
            if random.random() > 0.5:
                _face_frame_aug = self.flip_frames(_face_frame_aug, horizontal=True)
            
            # Random vertical flip (optional)
            if random.random() > 0.5:
                _face_frame_aug = self.flip_frames(_face_frame_aug, horizontal=False)


        if self.if_bg:
            
            if self.preload:
                _bg_frame_batch = self.preloaded_bg_images[_key][start:start+self.seq_length*num_batch]
            else:
                _bg_frame_batch = preload_frames(self.preloaded_bg_images[_key][start:start+self.seq_length*num_batch], self.transform_face, gray=self.gray)
            # _bg_frame_batch = self.preloaded_bg_images[_key][start:start+self.seq_length*num_batch]
            _bg_frame = torch.stack(_bg_frame_batch).transpose(0, 1)

        _ppg = torch.FloatTensor(self.subject_GT_PPG[_key][start:start+self.seq_length*num_batch])
        #print(f"{_face_frame.shape=}", f"{_face_frame_aug=}", f"{_ppg.shape=}")
        return _face_frame, _face_frame_aug, _bg_frame, _ppg, _key


    def __len__(self):
        return len(self.all_keys)
    
    
    
def get_loader(_datasets, _seq_length, batch_size=1, shuffle=True, train=True, if_bg=True):
    
    _dataset = rPPG_Dataset(datasets=_datasets, 
                            seq_length=_seq_length,
                            train=train,
                            if_bg=if_bg)

    # num_id, num_domain = _dataset.get_id_domain_num()
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":
    
    from einops import rearrange
    import numpy as np
    
    def saveFrames(frames):
        
        os.makedirs("test", exist_ok=True)
        # save tensor to images
        for i in range(frames.shape[2]):
            img = frames[0, :, i, :, :]
            # print(img.shape)
            # print(torch.max(img), torch.min(img))
            
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"test/{i}.jpg", img)

    
    train_loader = get_loader(_datasets=list("P"),
                              _seq_length=10*30,
                              batch_size=3,
                              train=True,
                              if_bg=False)
    test_loader = get_loader(_datasets=list("P"),
                              _seq_length=10*30,
                              batch_size=1,
                              train=False,
                              if_bg=False)

    print(f"{len(train_loader)=}")
    for step, (face_frames, face_frames_aug, bg_frames, ppg_label, subjects) in enumerate(train_loader):
        print(f"{face_frames.shape=}")
        print(f"{face_frames_aug.shape=}")
        print(f"{ppg_label.shape=}")
        print(subjects)
        # saveFrames(face_frames)

        # break
        
    
    print(f"{len(test_loader)=}")
    for step, (face_frames, face_frames_aug, bg_frames, ppg_label, subjects) in enumerate(test_loader):
        print(f"{face_frames.shape=}")
        print(f"{face_frames_aug=}")
        print(f"{ppg_label.shape=}")
        print(subjects)
        # saveFrames(face_frames)
        break
