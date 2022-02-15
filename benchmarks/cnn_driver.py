from data_utils import FramePairDataset
import torch
from torch.utils.data import Dataset, DataLoader

# def train():

# def eval():


# batch_size = 
# test_split = 
# shuffle =


# dataset = FramePairDataset(path_to_data)

# # Will need a different type of split/sampling to handle imbalanced classes? 
# test_size = int(test_split * dataset)
# train_size = len(dataset) - test_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))


# train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)