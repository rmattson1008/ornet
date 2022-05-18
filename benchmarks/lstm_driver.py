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

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import os
import math


def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    lr = args.lr
    epochs = args.epochs
    optimizer = Adam(model.parameters())
    # optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    model.to(device)
    for epoch in range(epochs):
        # print("Epoch", epoch + 1)
        model.train()
        training_loss = 0.0

        for i, data in enumerate(train_dataloader):
            # print("training on", i)
            # if i % 30 == 0:

            # get the inputs; data is a list of [inputs, labels]
            # print(data.shape)
            inputs, labels = get_augmented_batch(data)
            # print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()  # shouldn't stay on this step.
            # print("input shape", inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs.shape)
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


def get_augmented_batch(data):
    inputs, labels = data
    new_images = inputs.clone().detach()
    new_labels = labels.clone().detach()
    A_transforms = [
        [A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1), ToTensorV2()],
        [A.Transpose(p=1), ToTensorV2()],
        [A.Blur(blur_limit=7, always_apply=True), ToTensorV2()]
        # [A.RandomShadow((0,0,1,1), 5, 10, 3, p=1), ToTensorV2()],
    ]

    for t in A_transforms:
        t = A.Compose(t)
        # extend labels by one original batch
        new_labels = torch.cat((new_labels, labels))

        for sample in inputs:
            aug = [t(image=channel.numpy())["image"]
                   for frame in sample
                   for channel in frame]
            # print(len(aug))
            # print(aug[0].shape)
            aug = torch.stack(aug, dim=1)
            aug = aug.unsqueeze(2)
            # print("###")
            new_images = torch.cat((new_images, aug))
            # print(new_images.shape)

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
        inputs.size(0) + inputs.size(0)

    return new_images, new_labels


def get_dataloaders(args, accept_list):
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
        args.input_dir, accept_list, num_to_sample=args.sequence, class_types=args.classes, transform=transform)  # nt working :/

    print("dataset", len(dataset))

    # don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = math.floor(train_split * len(dataset))
    test_size = math.ceil((len(dataset) - train_size) / 2)
    val_size = math.floor((len(dataset) - train_size) / 2)
    train_size, test_size, val_size = int(
        train_size), int(test_size), int(val_size)

    assert train_size + val_size + test_size == len(dataset)
    # print("Train_size", train_size)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.weighted_samples:
        sampler = get_weighted_sampler(train_dataset, dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size)

    # print("Train_size", len(train_dataloader))
    return train_dataloader, test_dataloader, val_dataloader


def get_weighted_sampler(train_dataset, dataset):
    """
    Weighted Random Sampling
    One way to fight class imbalance
    """
    indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in indices]  # Messed up this
    train_dataset_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights = weights[y_train]

    sampler = WeightedRandomSampler(weights, len(train_dataset))

    return sampler


if __name__ == "__main__":
    print("Main")

    # going to use the same parser manager for cnn and rnn
    args, _ = make_parser()

    # num_lstm_layers = 10  # too many?
    # lstm_hidden_size = -1 # if we decide to have a different hidden space size then must add param to model

    model = CNN_LSTM(args.sequence)

    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'


    # device = 'cpu'
    print("device", device)



    """
    we have more segmented cell videos saved on logan then intermediates.
    best to rerun ornet on raw data, but till discrepancy is resolved this
    will result in using the 114 samples that Neelima used in the last scipy submit
    class balance 29, 31, 54
    """
    path_to_intermediates = "/data/ornet/gmm_intermediates"
    accept_list = []
    for subdir in args.classes:
        path = os.path.join(path_to_intermediates, subdir)
        for file in os.listdir(path):
            if 'normalized' in file:
                accept_list.append(file.split(".")[0])

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args, accept_list)
    # print("Train_size", len(train_dataloader))

    
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)

    # if not args.train:
    #     try:
    #         saved_state = torch.load(args.save)
    #         model.load_state_dict(saved_state)
    #         # how to check soemthing happened..
    #         print(type(model))
    #     except:
    #         # please exit
    #         print(
    #             "Please provide the path to an existing model state using --save \"<path>\" or train a new one with --train")
    #         exit()

    if args.test:
        print("Testing")
        test(args, model, test_dataloader, device=device)

# TODO - frame sampling with different batch sizes
