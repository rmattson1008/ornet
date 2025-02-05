import os
import unittest
import sys
import torch
import numpy as np
from torchvision import transforms

# trying to get the bechnmarks module to be visible to this file... 
# probably missing some concept but this works if you are working in your /ornet dir
curr_dir = os.getcwd()
sys.path.insert(0, curr_dir)
from baselines.data_utils import RoiTransform, FramePairDataset
from baselines.models import BaseCNN

data_path = '/data/ornet/single_cells_cnns/'
class_types = ['control', 'mdivi', 'llo']
roi_crop_size = (28,28)
roi_transform = RoiTransform(window_size=roi_crop_size)
scale_transform = transforms.Resize(size=28)
whitelist = []
for subdir in class_types:
    path = os.path.join("/data/ornet/gmm_intermediates/", subdir)
    files = os.listdir(path)
    whitelist.extend([x.split(".")[0] for x in files if 'normalized' in x])


model1 = BaseCNN()

# datum = iter(dataloader)[0]

class TestData(unittest.TestCase):
    # well this shows I accidently uploaded a DS_Store file to logan...
    # def test_input_dir(self):
    #     subdirectories = os.listdir(data_path)
    #     self.assertEqual(len(subdirectories), len(class_types))
    #     print(f"Using data from {len(subdirectories)} classes: {subdirectories}")

    def test_input_data(self):
        correct_vid_type = np.ndarray((200,512,512)) 
        for class_type in class_types:
            sub_dir = os.path.join(data_path, class_type)
            files = os.listdir(sub_dir)
            self.assertGreater(len(files), 0)
            for file_name in files:  
                    if 'normalized' in file_name:
                        vid_path = os.path.join(sub_dir, file_name)
                        try:
                            vid = np.load(vid_path)
                            self.assertEqual(type(vid), type(correct_vid_type))
                        except:
                            self.assertTrue(False)
                        self.assertTrue(True) # will this exit a loop

    def test_dataset(self):
        dataset = FramePairDataset(data_path, whitelist)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        correct_shape = torch.empty((1,2,512,512))
        for sample, label in dataloader:
            self.assertEqual(sample.shape, correct_shape.shape)

    def test_roi(self):
        dataset = FramePairDataset(data_path, whitelist, transform=roi_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        correct_shape = torch.empty((1,2,28,28))
        for sample, label in dataloader:
            roi = roi_transform(sample)
            self.assertEqual(roi.shape, correct_shape.shape)
        # calculate crop location by hand??
        # how is it any better than using the roi function

    # def test_dataloader(self):
    #     self.assertGreater(len(dataloader), 0)

    # def test_model(self):
        
    def test_feature_hook(self):
        dataset = FramePairDataset(data_path, whitelist, transform=scale_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        correct = torch.empty((1,10))
        for datum in dataloader:
            inputs, label = datum
            break

        inputs = inputs.float()
        # print(type(inputs))
        pred, features = model1(inputs)
        self.assertEqual(features["embedding10"].shape, correct.shape)


    # Definitely check the partitioning of datasets...
    # test each training and eval function???? test parser too? idk man idk
    # that models have hooks and subsequent representations are right

if __name__ == '__main__':
    unittest.main()    