import numpy as np
from matplotlib import pyplot as plt
import os

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.functional as TF
import cv2
import imageio


#kinda slow, 4-ish seconds to read in each video in a dataloader. 
class DynamicVids(Dataset):
    def __init__(self, path_to_folder, class_types=['control', 'mdivi', 'llo'], transform=None):
        """
        Initializes dataset by finding paths to all videos
        """
        self.targets= []
        self.vid_path = []
        self.transform = transform
        # self.transform_input = transform_input
        self.class_types = class_types

        path_to_folder = os.path.join(path_to_folder)
        print(path_to_folder)
        

        name_constraint = lambda x: 'normalized' in x
        for label in self.class_types:
            target =  self.class_types.index(label) #class 0,1,2
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            for file_name in files:
                    if  name_constraint(file_name):
                        self.vid_path.append(os.path.join(path, file_name))
                        self.targets.append(target)
        
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, idx):
        target_class = self.targets[idx]
        frames = vid_to_np_frames(self.vid_path[idx])
        # vid = np.load(self.vid_path[idx])
        # if num_frames == 2:
            # first_frame = vid[0]
            # last_frame = vid[-1] 
            # first_frame = torch.as_tensor((first_frame))
            # last_frame = torch.as_tensor((last_frame))
            # sample = torch.stack((first_frame, last_frame))


        frames = np.array((frames))
        sample = torch.as_tensor(frames)

        # add channel choice - arbitrary 1 for now but could redo with n many channels
        sample = sample.unsqueeze(1)
       
        if self.transform:
            sample = self.transform(sample)
            
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
    def __init__(self, path_to_folder, class_types=['control', 'mdivi', 'llo'], transform=None):
        self.targets= []
        self.vid_path = []
        self.transform = transform
        # self.transform_input = transform_input
        self.class_types = class_types

        print(path_to_folder)
        

        name_constraint = lambda x: 'normalized' in x
        for label in self.class_types:
            target =  self.class_types.index(label) #class 0,1,2
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            for file_name in files:
                    if  name_constraint(file_name):
                        self.vid_path.append(os.path.join(path, file_name))
                        self.targets.append(target)
        
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, idx):
        target_class = self.targets[idx]
        vid = np.load(self.vid_path[idx])
        first_frame = vid[0]
        last_frame = vid[-1] 

        first_frame = torch.as_tensor((first_frame))
        last_frame = torch.as_tensor((last_frame))

        # stack frames as channels, x shape is [2, img_width, img_height]
        sample = torch.stack((first_frame, last_frame))

       
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target_class


#TODO is this batchable, I think it is
# TODO, return some indication that image is empty. 
    #TODO - if it ends up empty, did protein move or dissapear?? how much do we handle at the moment
#TODO also this is kinda slow if that becomes a problem.
class RoiTransform:
    """Pick a ROI based on pixel intensity. Choose the same coordinates for both/all frames"""

    def __init__(self, window_size=(28,28), kernel_size=5, use_first_frame_values=True):
        """
        window_size = (width, height) of cropped section  
        kernel_size = size of gaussian kernel used in selecting peak pixel values  
        use_first_frame_values = if true, set ROI based off first frame, if false, set ROI based off last frame
        """
        self.width = window_size[0]
        self.height = window_size[1]
        self.kernel_size = kernel_size
        self.use_first_frame = use_first_frame_values

    def __call__(self, sample):
        """
        sample is [n_channels, img_width, img_height]
        """

        first_frame, last_frame = sample.split(1, dim=0)


        if self.use_first_frame:
            img_blurred = TF.gaussian_blur(first_frame, kernel_size=self.kernel_size) # TODO setting?
        else:
            img_blurred = TF.gaussian_blur(last_frame, kernel_size=self.kernel_size) # TODO setting?

        # used the topk function in case we later decide to take multiple peaks
        values, indices = torch.topk(img_blurred.flatten(), k=1)

        """ 
        Need to address the empty images (no mitochondria pictured) before train/test stage
        """
        # If first frame is empty, check the last frame for a signal
        if values[0] == 0.0:
            img_blurred = TF.gaussian_blur(last_frame, kernel_size=self.kernel_size) # TODO setting?
            values, indices = torch.topk(img_blurred.flatten(), k=1)
            if values[0] == 0.0:
                print("NO DATA: maximum pixel value is {}, but this sample will still be trained on".format(values[0]))

        np_idx = np.array(np.unravel_index(indices.numpy(), img_blurred.shape))

        top = int(np_idx[1] - 1/2 * self.width)
        left = int(np_idx[2] - 1/2 * self.height)
        assert first_frame.shape == img_blurred.shape

        # The crop is taken from original, unblurred image
        first_cropped = torchvision.transforms.functional.crop(first_frame, top, left, self.width, self.height)
        last_cropped = torchvision.transforms.functional.crop(last_frame, top, left, self.width, self.height)
        cropped = torch.cat((first_cropped, last_cropped), 0)
        # print("DEBUG: roi input size", cropped.size())

        return cropped
