import numpy as np
from matplotlib import pyplot as plt
import os

import torch
from torch.utils.data import Dataset



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
        # do we need to load the whole video to get the 2 frames
        target_class = self.targets[idx]
        vid = np.load(self.vid_path[idx])
        first_frame = vid[0]
        last_frame = vid[-1] 
        # assert u loaded the right thing? eh

        # ROI selection and such
        if self.transform is not None:
            first_frame = self.transform_image(first_frame)
            last_frame = self.transform_image(last_frame)
        

        # To be discussed
        # concept - should we take the difference, not the concatenation? 
        # implement - should this be done in a torch.transform?
        sample = np.concatenate((first_frame, last_frame))
        sample = torch.from_numpy(sample)

        return sample, target_class



