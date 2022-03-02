from data_utils import FramePairDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

def train(epoch):
    model.train()
    running_loss = 0.0
    print("== epoch", epoch, "==")
    count = 0

    # TODO - GPU compliant
        
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label = data
        inputs = inputs.float() # shouldn't stay on this step. 
        # print(label)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        # TODO - use val loss
        running_loss += loss.item()
        if i % 20 == 19:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0
            print(inputs.shape)

# def eval():
    # with torch.no_grad():
        # for data in test_dataloader:
        #     images, labels = data
        #     images = images.float()
        #     # calculate outputs by running images through the network
        #     outputs = model(images)
        #     _, predicted = torch.max(outputs.data, 1)


batch_size = 1
train_split = .8
# shuffle =
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=28)])
# TODO - normalize??
dataset = FramePairDataset(path_to_data, transform=transform)
dataset = FramePairDataset("../../../ornet-data/ornet-outputs/gray-frame-pairs/", transform=transform)

## don't think this is the way to do this... needs even percent split to work
train_size = int(train_split * len(dataset))
test_size = int((len(dataset) - train_size) / 2)
val_size = int((len(dataset) - train_size) / 2)

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
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

###
# run a simple model, no gpu no validation
#
####
model = Model_0()
# optimizer = Adam(model.parameters(), lr=0.07)
optimizer = SGD(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()
num_epochs = 5
for i in range(num_epochs):
    train(i)