import numpy as np
from matplotlib import pyplot as plt
import os

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.functional as TF
import cv2
import imageio
import math


#kinda slow, 4-ish seconds to read in each video in a dataloader. 
class DynamicVids(Dataset):
    def __init__(self, path_to_folder, accept_list, num_to_sample=10, class_types=['control', 'mdivi', 'llo'], transform=None):
        """
        Initializes dataset by finding paths to all videos
        """
        self.targets= []
        self.vid_path = []
        self.transform = transform
        # self.transform_input = transform_input
        self.class_types = class_types
        self.num_to_sample = num_to_sample

        path_to_folder = os.path.join(path_to_folder)
        print(path_to_folder)
        

        for label in self.class_types:
            target =  self.class_types.index(label) #class 0,1,2
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            for file_name in files:
                    if  file_name.split(".")[0] in accept_list:
                        self.vid_path.append(os.path.join(path, file_name))
                        self.targets.append(target)
        
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, idx):
        target_class = self.targets[idx]
        # frames = vid_to_np_frames(self.vid_path[idx])
        frames = np.load(self.vid_path[idx])
        # if frames.shape < 100:

        # don't know how to handle subsampling from variable length vids.
        #maybe better to just use random choice, 
        assert self.num_to_sample <= frames.shape[0]
        assert self.num_to_sample <= 50 # frame selection is just gonna be brittle
        if frames.shape[0] == 200:
            step = math.ceil(frames.shape[0] / self.num_to_sample)
            # step = round(frames.shape[0] / self.num_to_sample)
            frames = frames[0:-1:step]
        elif frames.shape[0] < 200:
            # step = math.round(frames.shape[0] / self.num_to_sample)
            step = math.floor(frames.shape[0] / self.num_to_sample)
            frames = frames[0:-1:step]
            # truncate some extras that show up due to floating point div
            frames = frames[:self.num_to_sample]
        else:
            print("Why's your data got more than 200 frames I didn't train for this")
        assert frames.shape[0] == self.num_to_sample



        # vid = np.load(self.vid_path[idx])
        # if num_frames == 2:
            # first_frame = vid[0]
            # last_frame = vid[-1] 
            # first_frame = torch.as_tensor((first_frame))
            # last_frame = torch.as_tensor((last_frame))
            # sample = torch.stack((first_frame, last_frame))


        # frames = np.array((frames))
        sample = torch.as_tensor(frames)

        # add channel choice - arbitrary 1 for now but could redo with n many channels
        sample = sample.unsqueeze(1)
        # print(sample.shape)
        # sample = torch.reshape(sample, (11,3,512,512))
        # print(sample.shape)
       
        if self.transform:
            sample = self.transform(sample)
            
        # print(sample.shape)
        sample = torch.as_tensor(sample)
        sample = sample.float()
        return sample, target_class


def vid_to_np_frames(vid_path):
    frames = []
    reader = imageio.get_reader(vid_path)
    for frame in reader:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    reader.close()
    return frames



###
# The dataset initializes with the file path to frame pairs
# and loads the image when get image is called. Did this back when planning to take frames 
# from entire video. Now the data is stored as as only 2 frames, so not really a necessary step
###
class FramePairDataset(Dataset):
    """
    dataset class for first and last frames of cell videos. 
    """
    def __init__(self, path_to_folder, accept_list=[], class_types=['control', 'mdivi', 'llo'], transform=None, augmentations=None):
        """
        path_to_folder: directory containing subfolders of class instances
        accept_list: list of all file names that should be included in the dataset. Contrains unwanted samples.
        class_types: list of possible class names
        transform: torch type transform, typically for base transformation
        augmentation: ablumentations type transforms, for augmenting/expanding data set
        """
        self.targets= []
        self.vid_path = []
        self.transform = transform
        # self.aug = augmentations
        self.class_types = class_types

        print(path_to_folder)
        

        for label in self.class_types:
            target =  self.class_types.index(label) #class 0,1,2
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            for file_name in files:
                    # if 'normalized' in file_name:
                    if file_name.split(".")[0] in accept_list:
                        self.vid_path.append(os.path.join(path, file_name))
                        self.targets.append(target)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target_class = self.targets[idx]
        vid = np.load(self.vid_path[idx])

        sample = vid[0:2]
        assert sample.shape == (2,512,512) 

        # TODO - will probably cause error during
        s = torch.as_tensor(sample)
        if self.transform:
            # print("calling transform")
            sample = self.transform(s)
            # sample = self.transform(sample)
        
        sample = torch.as_tensor(sample)
        sample = sample.float()
        return sample, target_class

        #### is it???



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