import os
import unittest
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/rachel/ornet/')
from benchmarks.data_utils import RoiTransform, FramePairDataset

data_path = '/data/ornet/single_cells_cnns'
class_types = ['control', 'mdivi', 'llo']
roi_crop_size = (28,28)
roi_transform = RoiTransform(window_size=roi_crop_size)
dataset = FramePairDataset(data_path)

class TestData(unittest.TestCase):
    # well this shows I accidently uploaded a DS_Store file to logan...
    def test_input_dir(self):
        subdirectories = os.listdir(data_path)
        self.assertEqual(len(subdirectories), len(class_types))
        print(f"Using data from {len(subdirectories)} classes: {subdirectories}")

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
        correct_shape = torch.empty((2,512,512))
        for sample, label in dataset:
            self.assertEqual(sample.shape, correct_shape.shape)

    def test_roi(self):
        correct_shape = torch.empty((2,28,28))
        for sample, label in dataset:
            roi = roi_transform(sample)
            self.assertEqual(roi.shape, correct_shape.shape)
        # calculate crop location by hand??
        # how is it any better than using the roi function

    # Definitely check the partitioning of datasets...
    # test each training and eval function???? test parser too? idk man idk

if __name__ == '__main__':
    unittest.main()    