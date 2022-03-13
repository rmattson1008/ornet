import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


from data_utils import FramePairDataset
from models import Model_0
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser


def train(args, model, device, train_dataloader):
    lr = args.lr
    epochs = args.epochs
    # optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    model.train()
    # model.to(device)

    running_loss = 0.0

    for epoch in range(epochs):  
        print("== epoch", epoch, "==")
        for i, data in enumerate(train_dataloader, 0): #???? where/how is batchsize handled
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float() # shouldn't stay on this step. 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # TODO - use val loss
            running_loss += loss.item()
            if i % 20 == 19:   
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
                print(inputs.shape)
    return

def eval(show_plots=False):
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.float()
            # calculate outputs by running images through the network
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    # args.classlist()
    classes = ['control', 'mdivi', 'llo']
    cm = confusion_matrix(labels, predicted, classes)
    # accuracy = (labels == predicted).sum() / len(labels)
    accuracy = np.sum(labels == predicted) / len(labels)
    print("Accuracy:", accuracy)
    # false pos. false negative... CM... three

    if show_plots:
        print(cm) #TODO - make pretty

    #what tf else we doing
    return


# # batch_size = 1
# # train_split = .8
# # shuffle =
# transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=28)])
# # TODO - normalize??
# dataset = FramePairDataset(path_to_data, transform=transform)
# dataset = FramePairDataset("../../../ornet-data/ornet-outputs/gray-frame-pairs/", transform=transform)
# classes = FramePairDataset.class_types #TODO - clean

# ## don't think this is the way to do this... needs even percent split to work
# train_size = int(train_split * len(dataset))
# test_size = int((len(dataset) - train_size) / 2)
# val_size = int((len(dataset) - train_size) / 2)

# # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
# # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)



# return array with the class weight for each instance in subset
def get_weights(indices):
    """
    helper function for weighted sampling
    """
    y_train = [dataset.targets[i] for i in indices]
    train_dataset_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights = weights[y_train]
    return weights

def get_weighted_sampling(train_dataset):
    ###
    # Weighted Random Sampling
    # One way to fight class imbalance
    ###
    train_indices = train_dataset.indices
    weights = get_weights(train_indices)
    sampler = WeightedRandomSampler(weights, len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_dataloader


def get_dataloaders(args):
    transform = transforms.Compose([transforms.Resize(size=28)])
    # TODO - normalize??
    dataset = FramePairDataset(args.input_dir, class_types=args.classes, transform=transform)

    
    
    ## don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = int(train_split * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = int((len(dataset) - train_size) / 2)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.weighted_sample:
        train_dataloader = get_weighted_sampling(train_dataset)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader


if __name__ == "__main__":
    # model
    model = Model_0()

    args, _ = make_parser()
    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)

    train(args, model, device, train_dataloader)

   

    # eval