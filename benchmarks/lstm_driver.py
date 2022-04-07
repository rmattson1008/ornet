import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


from data_utils import DynamicVids, RoiTransform
from models import CNN_LSTM
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser 

from matplotlib import pyplot as plt

def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    lr = args.lr
    epochs = args.epochs
    # optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()


    model.to(device)
    for epoch in range(epochs):  
        print("Epoch", epoch)
        model.train()
        training_loss = 0.0

        for data in train_dataloader: 
            # get the inputs; data is a list of [inputs, labels]
            # print(data.shape)
            inputs, labels = data[0].to(device), data[1].to(device)
            print(labels)
            inputs = inputs.float() # shouldn't stay on this step.
            print("input shape", inputs.shape)
         

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

    return



def get_dataloaders(args, display_images=False):
    """
    Access ornet dataset, apply any necessary transformations to images, 
    split into train/test/validate, and return dataloaders
    """

    #
    if args.roi:
        print("Using ROI inputs")
        transform = transforms.Compose([RoiTransform(window_size=(28,28))])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=28)])
    dataset = DynamicVids(args.input_dir, class_types=args.classes, transform=transform) # nt working :/ 

    print("dataset", len(dataset))

    ## don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = int(train_split * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = int((len(dataset) - train_size) / 2)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # if args.weighted_samples:
    #     sampler = get_weighted_sampler(train_dataset, dataset)
    #     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    # else:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader




if __name__ == "__main__":
    print("Main")
    # going to use the same parser manager for cnn and rnn
    args, _ = make_parser()

    num_lstm_layers = 200 #too many?
    # lstm_hidden_size = -1 # if we decide to have a different hidden space size then must add param to model

    model = CNN_LSTM(num_lstm_layers)

    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    # device = 'cpu'
    print("device", device)
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)
    print("Train_size", len(train_dataloader))
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)
    
    # if args.test:
    #     #TODO - if no model, load saved model
    #     print("Testing")
    #     test(args, model, test_dataloader)

