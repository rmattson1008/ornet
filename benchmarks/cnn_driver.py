import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


from data_utils import FramePairDataset, RoiTransform
from models import Model_0
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
        model.train()
        training_loss = 0.0
        # running_loss = 0.0
        for data in train_dataloader: 
            # get the inputs; data is a list of [inputs, labels]
            # print(data.shape)
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.float() # shouldn't stay on this step. 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            # # print statistics
            # running_loss += loss.item()
            # if i % 20 == 19:   
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            #     running_loss = 0.0 # weird to do like this idk. 
            #     print(inputs.shape)


        model.eval()
        valid_loss = 0.0
        for inputs, labels in val_dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float() # shouldn't stay on this step. 

            pred = model(inputs)
            loss = criterion(pred, labels)
            valid_loss += loss.item()


        print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
             \t\t Validation Loss: {valid_loss / len(val_dataloader )}')


        #TODO - stop training when Val drops?

    return


def test(args, model, show_plots=True, device='cpu'):
    model.eval() #is necessary? 

    #TODO - should this be on gpu
    with torch.no_grad():
        y_true = torch.tensor([])
        y_pred = torch.tensor([])
        for data in test_dataloader:
            images, labels = data
            
            images = images.float()
            # calculate outputs by running images through the network
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            #bad approach?
            y_pred = torch.cat((y_pred, predicted), 0) 
            y_true = torch.cat((y_true, labels), 0) 

    # args.classlist()
    cm = confusion_matrix(y_true, y_pred)

    assert len(y_pred) == len(y_true)
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print("Accuracy:", accuracy)

    if show_plots:
        print(args.classes)
        print(cm) #TODO - make pretty

    #what tf else we doing
    return


def get_weighted_sampler(train_dataset, dataset):
    """
    Weighted Random Sampling
    One way to fight class imbalance
    """
    indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in indices] # Messed up this
    train_dataset_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights = weights[y_train]

    sampler = WeightedRandomSampler(weights, len(train_dataset))
    
    return sampler


def get_dataloaders(args, display_images=False):
    """
    Access ornet dataset, apply any necessary transformations to images, 
    split into train/test/validate, and return dataloaders
    """


    if args.roi:
        print("Using ROI inputs")
        transform = transforms.Compose([RoiTransform(window_size=(28,28))])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=28)])
    # TODO - normalize??
    dataset = FramePairDataset(args.input_dir, class_types=args.classes, transform=transform)

    
    
    if display_images:
        # This code is here if you are like me and need to see to believe your data exists
        for i, ((img, _), __) in enumerate(dataset):
            if i % 20 == 0:
                plt.imshow(img)
                plt.show()
    
    ## don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = int(train_split * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = int((len(dataset) - train_size) / 2)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.weighted_samples:
        sampler = get_weighted_sampler(train_dataset, dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader


if __name__ == "__main__":
    model = Model_0()

    args, _ = make_parser()
    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)
    
    if args.test:
        #TODO - if no model, load saved model
        print("Testing")
        test(args, model, test_dataloader)

   #TODO - save models?
