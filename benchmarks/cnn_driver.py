from data_utils import FramePairDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# def train():

# def eval():



# batch_size = 
# train_split = 
# shuffle =
dataset = FramePairDataset(path_to_data)

## don't think this is the way to do this... needs even percent split to work
train_size = int(train_split * len(dataset))
test_size = int((len(dataset) - train_size) / 2)
val_size = int((len(dataset) - train_size) / 2)

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


###
# Weighted Random Sampling
# One way to fight class imbalance
# pretty sure we only need it in training?
###
train_indices = train_dataset.indices

# return array with the class weight for each instance in subset
def get_weights(indices):
    y_train = [dataset.targets[i] for i in indices]
    train_dataset_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights = weights[y_train]
    return weights

weights = get_weights(train_indices)
sampler = WeightedRandomSampler(weights, len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)