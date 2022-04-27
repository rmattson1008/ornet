import numpy as np
from matplotlib import pyplot as plt
import os

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.functional as TF
import cv2




###
# The dataset initializes with the file path to frame pairs
# and loads the image when get image is called. Did this back when planning to take frames 
# from entire video. Now the data is stored as as only 2 frames, so not really a necessary step
###
class FramePairDataset(Dataset):
    def __init__(self, path_to_folder, class_types=['control', 'mdivi', 'llo'], transform=None):
        self.targets= []
        self.vid_path = []
        self.transform = transform
        self.class_types = class_types

        print(path_to_folder)
        

        for label in self.class_types:
            target =  self.class_types.index(label) #class 0,1,2
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            for file_name in files:
                    if 'normalized' in file_name:
                        self.vid_path.append(os.path.join(path, file_name))
                        self.targets.append(target)
        
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, idx):
        target_class = self.targets[idx]
        vid = np.load(self.vid_path[idx])

        sample = vid[0:2]
        assert sample.shape == (2,512,512) 

        if self.transform:
            try:
                s = torch.as_tensor(sample)
                sample = self.transform(s)
            except KeyError: 
                #disgusting, trying to get albumentations tranforms to work over channels...
                # idk they have channel specific transforms so this can't be neccesary
                # frame1 = cv2.cvtColor(sample[0], cv2.COLOR_GRAY2RGB)
                # frame2 = cv2.cvtColor(sample[1], cv2.COLOR_GRAY2RGB)
                frame1 = sample[0]
                frame2 = sample[1]
                frame1 = self.transform(image=frame1)['image']
                # frame1 = self.transform(image=frame1)
                frame2 = self.transform(image=frame2)['image']
                # frame2 = self.transform(image=frame2)
                sample = np.concatenate((frame1,frame2))
                assert sample.shape == (2,512,512)

        sample = torch.as_tensor(sample)
        return sample, target_class



class RoiTransform:
    """Pick a ROI based on pixel intensity. Choose the same coordinates for both/all frames"""

    def __init__(self, window_size=(28,28), kernel_size=5, use_frame_idx=0):
        """
        window_size = (width, height) of cropped section  
        kernel_size = size of gaussian kernel used in selecting peak pixel values  
        use_frame_idx = frame whose values determines the crop. 0 for first and 1 for last.
        """
        self.crop_width = window_size[0]
        self.crop_height = window_size[1]
        self.kernel_size = kernel_size
        self.preferred_frame_idx = use_frame_idx

    def __call__(self, sample):
        """
        sample is [n_channels, img_width, img_height]
        """
        # I think there is experimental value on basing the ROI off the first frame versus the last frame
        # so building in that choice
        use_frame = self.preferred_frame_idx
        
        blurred_images = TF.gaussian_blur(sample, kernel_size=self.kernel_size)

        # used the topk function in case we later decide to take multiple peaks
        values, indices = torch.topk(blurred_images[use_frame].flatten(), k=1)

        # when there's no values in the frame considered, we check the other one
        # just a design choice, we might decide we don't want to be flipping back and forth willy nilly
        if values[0] <= 0:
            use_frame = 1 if use_frame == 0 else 0 #flip preference
            values, indices = torch.topk(blurred_images[use_frame].flatten(), k=1)
            if values[0] == 0.0:
                print(f"NO DATA: maximum pixel value is {values[0]}, but this sample will still be trained on")
                #TODO - handle better
                #TODO - if it ends up empty, did protein move or dissapear?? how much do we handle for baseline

        # Basically we want to find the coordinates of a bounding box around relevant max pixel.
        np_idx = np.array(np.unravel_index(indices.numpy(), blurred_images[use_frame].shape))
        top = int(np_idx[0] - 1/2 * self.crop_width)
        left = int(np_idx[1] - 1/2 * self.crop_height)

        # The crop is taken from original, unblurred image
        cropped = torchvision.transforms.functional.crop(sample, top, left, self.crop_width, self.crop_height)
        # print("DEBUG: roi input size", cropped.size())

        return cropped