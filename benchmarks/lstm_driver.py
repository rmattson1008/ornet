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

        for i, data in enumerate(train_dataloader):
            # if i % 30 == 0:

            # get the inputs; data is a list of [inputs, labels]
            # print(data.shape)
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(labels)
            inputs = inputs.float()  # shouldn't stay on this step.
            # print("input shape", inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        for inputs, labels in val_dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()  # shouldn't stay on this step.

            logits, hidden_state = model(inputs)
            # print(logits.shape)
            _, pred = torch.max(logits, 1)
            loss = criterion(logits, labels)
            valid_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
             \t\t Validation Loss: {valid_loss / len(val_dataloader )}')

        # TODO - stop training when Val drops? val score is wack right now

    if args.save:
        print(args.save)
        print("Saving model")
        torch.save(model.state_dict(), args.save)

    return


def test(args,  model, test_dataloader, show_plots=True, device='cpu'):
    model.eval()  # is necessary?

    with torch.no_grad():
        y_true = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            # calculate outputs by running images through the network
            logits, _ = model(images)
            # print(logits.shape)
            _, predicted = torch.max(logits, 1)
            # print(predicted.shape)
            # bad approach?
            y_pred = torch.cat((y_pred, predicted), 0)
            y_true = torch.cat((y_true, labels), 0)

    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())

    assert len(y_pred) == len(y_true)
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print("Accuracy:", accuracy)

    if show_plots:
        print(args.classes)
        print(cm)  # TODO - make pretty

    return


def get_dataloaders(args, display_images=False):
    """
    Access ornet dataset, apply any necessary transformations to images, 
    split into train/test/validate, and return dataloaders
    """

    #
    if args.roi:
        print("Using ROI inputs")
        transform = transforms.Compose([RoiTransform(window_size=(28, 28))])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=28)])
    dataset = DynamicVids(
        args.input_dir, class_types=args.classes, transform=transform)  # nt working :/

    print("dataset", len(dataset))

    # don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = int(train_split * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = int((len(dataset) - train_size) / 2)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
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

    num_lstm_layers = 200  # too many?
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

    if args.test:
        if not args.train:
            try:
                saved_state = torch.load(args.save)
                model.load_state_dict(saved_state)
                # how to check soemthing happened..
                print(type(model))
            except:
                # please exit
                print(
                    "Please provide the path to an existing model state using --save \"<path>\" or train a new one with --train")
                exit()
        print("Testing")
        test(args, model, test_dataloader, device=device)
