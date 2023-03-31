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
import random

"""All pre-trained models expect input images normalized in the same way, 
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
where H and W are expected to be at least 224. The images have to be loaded 
in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
"""
class TimeChunks(Dataset):
    def __init__(self, path_to_folder, accept_list, frames_per_chunk=2 , step=1, class_types=['control', 'mdivi', 'llo'], transform=None, verbose=False, shuffle_chunks=True, random_wildtype_p=None):
        """
        Initializes dataset. For now samples all frames possible that is can. could be more selective (undersample "overrepresented")

        Each instance is  [T, H, W]: we have only one channel for each greyscale image so the time dimension is more important.

        path_to_folder: path to directory containing 3 subdirectories. Folder names should match the class_types list
        accept_list: list of file names to use, in order to ignore data samples we dropped in past studies (for comparison). no option to disregard this list yet
        frames_per_chunk: number of frames in each intended sample. 
        step: distance between each frame that is sampled. Ex. if  frames_per_chunk = 3 and step = 3, an instance will span 9 frames of the video. 


        """
        print("p", random_wildtype_p)
        random.seed(69)

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

        samples = []
        for label in self.class_types:
            print(label)
        # TODO this is wrong
            if label == "llo":
                target = 0
            elif label == "mdivi":
                target = 1
            elif label == "control":
                target = 2
            else:
                print("that didn't work. are you using the right data?")

            path = os.path.join(path_to_folder, label)
            files = os.listdir(path)
            
            for file_name in files:
                if  file_name.split(".")[0] in accept_list:
                    # print("label", label)

                    #read vid, get length, discard vid
                    path_to_vid = os.path.join(path, file_name)
                    num_frames = len(np.load(path_to_vid))

                    l = range(0,num_frames)
                    if target == 0:
                        c_indeces = l[50:-self.frames_per_chunk * self.step_size:self.frames_per_chunk * self.step_size]
                    elif target == 1:
                        c_indeces = l[0:-self.frames_per_chunk * self.step_size:self.frames_per_chunk * self.step_size]
                    elif target == 2:
                        # add control with random label to dataset with prob p
                        # this is not like label flipping because the label is set for the duration of training. At some point, the model will start overfitting and memorize which category each control sample belongs to.
                        if random_wildtype_p:
                            c_indeces = l[0:-self.frames_per_chunk * self.step_size:self.frames_per_chunk * self.step_size]
                            number_to_take =  round(random_wildtype_p * len(c_indeces))
                            # print("sampling", number_to_take, "elements from each control video")
                            c_indeces = random.sample(c_indeces, number_to_take)
                            choose = [0,1]
                            target = random.choice(choose) # with random label
                        else:
                            continue

                    else: #no associated target
                        print("SOMETHING WENT WRONG")
                    
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
        one_hot = {0:torch.tensor([1,0], dtype=torch.float), 1:torch.tensor([0,1],  dtype=torch.float)} #two brain cells
        
        try:
            path, start_idx = self.samples[idx]
        except(IndexError):
            print("trying to access index", idx)
            print("samples is length", len(self.samples))
        else:
            path, start_idx = self.samples[idx]
        end_index = start_idx + self.frames_per_chunk * self.step_size 
        
        # maybe clean up these if elses?
        if self.verbose:
            target = {'label':one_hot[self.targets[idx]],'vid_path': path, 'range':(start_idx, end_index), 'step': self.step_size }
        else:
            target = self.targets[idx]
            # print("target:",target)
            target=one_hot[target]
            # print("target:", target)

 
        frames = np.load(path)       
        sample = frames[start_idx:end_index:self.step_size]
    
        # if you want it to be a smooth transformation need to rearrange the ndarray to be (H x W x T) before transforms
        sample = np.transpose(sample, (1, 2, 0))
        if self.transform:
            sample = self.transform(sample)

        sample = sample.float()
      
        return sample, target
       


def get_accept_list(path_to_intermediates, classes) -> list:
    #TODO - this should become an arguement
    path_to_intermediates = "/mnt/data4TBa/ram13275/gmm_intermediates" 
    accept_list = []
    class_vector = []
    for subdir in classes:
            path = os.path.join(path_to_intermediates, subdir)
            files = os.listdir(path)
            lst = [x.split(".")[0]for x in files if 'normalized' in x]
            accept_list.extend(lst)
            class_vector.extend([subdir for l in lst])

            # mdivi1 :100
            # mdivi2 100:

    return accept_list, class_vector