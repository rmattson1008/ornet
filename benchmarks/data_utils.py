import numpy as np
from matplotlib import mathtext, pyplot as plt
import os

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.functional as TF
import cv2
import imageio
import math
from torchvision import transforms
import pandas as pd


"""All pre-trained models expect input images normalized in the same way, 
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
where H and W are expected to be at least 224. The images have to be loaded 
in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
#  so its ok if the most immediate input is not between 0 and one? why so many negatives? 
"""
class TimeChunks(Dataset):
    def __init__(self, path_to_folder, accept_list, frames_per_chunk=2 , step=1, class_types=['control', 'mdivi', 'llo'], transform=None, verbose=False, shuffle_chunks=True):
        """
        Initializes dataset. For now samples all frames possible that is can. could be more selective (undersample "overrepresented")

        Only samples from MDIVI and LLO right now. 

        Each instance is  [T, H, W]: we have only one channel for each greyscale image so the time dimension is more important.

        path_to_folder: path to directory containing 3 subdirectories. Folder names should match the class_types list
        accept_list: list of file names to use, in order to ignore data samples we dropped in past studies (for comparison). no option to disregard this list yet
        frames_per_chunk: number of frames in each intended sample. 
        step: distance between each frame that is sampled. Ex. if  frames_per_chunk = 3 and step = 3, an instance will span 9 frames of the video. 


        """
        # landing place for X,y
        self.targets = []
        # save tuple ("path", starting idk, step)
        self.samples = []
        self.shuffle_chunks = shuffle_chunks

        # self.vid_path = []
        # useful variables
        self.transform = transform
        # self.transform_input = transform_input
        self.class_types = class_types
        self.frames_per_chunk = frames_per_chunk
        self.step_size = step
        self.verbose = verbose

        path_to_folder = os.path.join(path_to_folder)
        # print(path_to_folder)
        
# TODO - trimmed llo, don't need separate folder just list of trimmed. 
        samples = []
        for label in self.class_types:
            # print("label", label)
        
            if label == "llo":
                target = 0
            elif label == "mdivi":
                target = 1
            else:
                continue

            # print("TARGET", target)
            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            
            for file_name in files:
                    if  file_name.split(".")[0] in accept_list:
                        #read vid, get length, discard vid
                        path_to_vid = os.path.join(path, file_name)
                        num_frames = len(np.load(path_to_vid))

                        l = range(0,num_frames)
                        if target == 0:
                            c_indeces = l[50:-self.frames_per_chunk * self.step_size:self.frames_per_chunk * self.step_size]
                        elif target == 1:
                            c_indeces = l[0:-self.frames_per_chunk * self.step_size:self.frames_per_chunk * self.step_size]
                        else:
                            print("FIX UR DATASET")


                        chunks = [(target, path_to_vid, start_idx) for start_idx in c_indeces]
                        samples.extend(chunks)



        arr = pd.DataFrame(samples)
        if self.shuffle_chunks:
            print("Randomly shuffling chunks to prevent neighbors coming from the same video")
            arr = arr.sample(frac=1, random_state=42)

        self.targets = arr[0].to_numpy()
        self.samples = arr[[1, 2]].to_numpy()            
        
    def __len__(self):
        return len(self.targets)

    def get_y_dist(self):
        return self.targets

    
    def __getitem__(self, idx):
        
        try:
            path, start_idx = self.samples[idx]
        except(IndexError):
            print("trying to access index", idx)
            print("samples is length", len(self.samples))
        else:
            path, start_idx = self.samples[idx]
        end_index = start_idx + self.frames_per_chunk * self.step_size 
        
        if self.verbose:
            target = {'label':self.targets[idx],'vid_path': path, 'range':(start_idx, end_index), 'step': self.step_size }
        else:
            target = self.targets[idx]

 
        frames = np.load(path)       
        sample = frames[start_idx:end_index:self.step_size]
    
        # if you want it to be a smooth transformation need to rearrange the ndarray to be (H x W x T) before transforms
        sample = np.transpose(sample, (1, 2, 0))
        if self.transform:
            sample = self.transform(sample)

        sample = sample.float()
      
        return sample, target
       


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
        # assert sample.shape == (2,512,512) 

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

def get_accept_list(path_to_intermediates, classes) -> list:
    path_to_intermediates = "/data/ornet/gmm_intermediates"
    accept_list = []
    class_vector = []
    for subdir in classes:
            path = os.path.join(path_to_intermediates, subdir)
            files = os.listdir(path)
            lst = [x.split(".")[0]for x in files if 'normalized' in x]
            if subdir != 'control':
                accept_list.extend(lst)
                class_vector.extend([subdir for l in lst])

            # mdivi1 :100
            # mdivi2 100:

    return accept_list, class_vector