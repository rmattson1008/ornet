import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torchvision
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import pickle

from data_utils import FramePairDataset, RoiTransform
from models import BaseCNN, VGG_Model, ResNet18, ResBlock
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from time import sleep

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter


def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    lr = args.lr
    epochs = args.epochs
    # optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    tb = SummaryWriter()

    train_losses = []
    val_losses = []

    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()
        training_loss = 0.0
        total_correct = 0.0
        valid_loss = 0.0
        for data in train_dataloader:
            # inputs, labels = get_augmented_batch(data)
            images, labels = data
            # print(images.dtype)

            inputs, labels = images.to(device), labels.to(device)
            # inputs = inputs.float()  # shouldn't stay on this step.
            # inputs = inputs.to(torch.float32)  # shouldn't stay on this step.
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.shape)
            # print(inputs.dtype)
            # print(inputs.shape)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            total_correct+= outputs.argmax(dim=1).eq(labels).sum().item()

        model.eval()
        
        for inputs, labels in val_dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.float()  # shouldn't stay on this step.
            pred, _ = model(inputs)
            loss = criterion(pred, labels)
            valid_loss += loss.item() 
            

        # print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
            # \t\t Validation Loss: {valid_loss / len(val_dataloader )}')

        val_loss_epoch = valid_loss / len(val_dataloader)
        train_loss_epoch = training_loss / len(train_dataloader)
        accuracy = total_correct / len(train_dataloader)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)

        total_correct = sum(pred == labels)

        #todo - save value over batch?
        tb.add_scalar("TrainLoss", training_loss, epoch)
        tb.add_scalar("ValLoss", valid_loss, epoch)
        # tb.add_scalar("Correct", total_correct, epoch)
        # print(accuracy.shape)
        tb.add_scalar("Accuracy", accuracy, epoch)
        # print(model.named_parameters())
        for name, weight in model.named_parameters():
            tb.add_histogram(name,weight, epoch)
            tb.add_histogram(f'{name}.grad',weight.grad, epoch)
        # print(np.min(val_losses))
        if val_loss_epoch == np.min(val_losses) and args.save_model:
            # print("Saving model")
            torch.save(
                {'epoch': epoch + 1,
                 'state_dict': model.state_dict()},
                args.save_model)

    plot_dict = {"train_loss": train_losses, "val_loss": val_losses}

    if args.save_losses:
        print("saving loss plots")
        path = args.save_losses
        with open(path, "wb") as tf:
            pickle.dump(plot_dict, tf)
    tb.close()
    return
    # todo - move this down and clean up args
    # if args.save_model:
    #     print(args.save_model)
    #     print("Saving model")
    #     torch.save(model.state_dict(), os.path.join(
    #         args.save_dir, args.save_model))

    # if args.save_losses:
    #     print("saving loss plots")
    #     path = "/home/rachel/ornet/losses/l2.pkl"
    #     # path=os.path.join(args.save_losses, args.save_model)
    #     plot_dict={"train_loss": train_losses, "val_loss": val_losses}
    #     with open(path, "wb") as tf:
    #         pickle.dump(plot_dict, tf)

    # return


def test(args, model, test_dataloader, show_plots=True, device='cpu'):
    model.eval()  # is necessary?

    with torch.no_grad():
        y_true = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            # calculate outputs by running images through the network
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
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


# TODO - some other weight to lessen the importance of llo in training.
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

    random.seed(73)
    for t in A_transforms:
        t = A.Compose(t)
        # extend labels by one original batch
        new_labels = torch.cat((new_labels, labels))

        for sample in inputs:
            aug = [t(image=channel.numpy())["image"]
                   for channel in sample]
            aug = torch.stack(aug, dim=1)
            new_images = torch.cat((new_images, aug))

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
        inputs.size(0) + inputs.size(0)

    return new_images, new_labels


def get_dataloaders(args, accept_list, resize=224):
    """
    Access ornet dataset, apply any necessary transformations to images,
    split into train/test/validate, and return dataloaders
    """

    if args.roi:
        print("Using ROI inputs")
        # default interpolation is bilinear, no idea if there is better choice
        transform = transforms.Compose(
            [RoiTransform(window_size=(28, 28)), transforms.Resize(size=resize)])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=resize)])
        # transform = []
        # print("!!!!! using 224x224")
        # transform = transforms.Compose([transforms.Resize(size=224)])
    # TODO - normalize??
    # dataset = FramePairDataset(args.input_dir, class_types=args.classes)
    dataset = FramePairDataset(
        args.input_dir, accept_list=accept_list, class_types=args.classes, transform=transform)
    # dataset = augment_dataset(dataset, transform)

    assert len(dataset) > 0

    # don't think this is the best way to do this... needs even percent split to work
    # brittle
    train_split = .8
    train_size = math.floor(train_split * len(dataset))
    test_size = math.ceil((len(dataset) - train_size) / 2)
    val_size = math.floor((len(dataset) - train_size) / 2)
    train_size, test_size, val_size = int(
        train_size), int(test_size), int(val_size)

    assert train_size + val_size + test_size == len(dataset)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed())
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(73))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # augment images
    # print(len(train_dataset))
    # train_dataset = augment_dataset(train_dataset, transform)
    # print(len(train_dataset))

    if args.weighted_samples:
        print("using weighted sampling")
        sampler = get_weighted_sampler(train_dataset, dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader


def get_deep_features(args, model, loader_dict, device="cpu"):
    print("Getting deep features")
    checkpoint = torch.load(args.save_model)
    model.load_state_dict(checkpoint['state_dict'])
    print("best performance at epoch", checkpoint['epoch'])

    model.eval()
    feature_dict = {}
    for (loader_name, loader) in loader_dict.items():
        frames = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs, features = model(inputs)

            labels = labels.cpu().detach().numpy()
            features = features['embeddings10'].cpu().detach().numpy()

            df = pd.DataFrame(features)
            df['label'] = labels
            frames.append(df)
        final_df = pd.concat(frames)
        feature_dict[loader_name] = final_df
        # save_path = args.save_features.split(".")[0] + name + "." + args.save_features.split(".")[1]

    with open(args.save_features, 'wb') as f:
        pickle.dump(feature_dict, f)

    return feature_dict



if __name__ == "__main__":

    args, _ = make_parser()
    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    device = torch.device(device)
    print(device)
    # tb = SummaryWriter()

    # throwing kitchen sink of deterministic p.
    # SGD is obv stochastic.... no way to seed? 
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(73)
    if device =='cuda':
        torch.cuda.manual_seed_all(73)

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
        files = os.listdir(path)
        accept_list.extend([x.split(".")[0]
                           for x in files if 'normalized' in x])

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        args, accept_list, resize=212)

    

    # model = BaseCNN()
    # model = VGG_Model()
    model = ResNet18(in_channels=2, resblock=ResBlock, outputs=3)



    # images, labels = next(iter(train_dataloader))
    # print("got image batch from laoder")
    # images = transforms.ToPILImage()(images)

    # # images = images.astype('uint8')
    # grid = torchvision.utils.make_grid(images)  
    # print("made grid")
    # tb.add_image("images", grid)
    # print("added image grid to tb")
    # print(type(images))
    # print(images.shape)
    # # images = images.numpy()
    # print(type(images))

    # images = transforms.ToPILImage()(images)
    # I dont think the two channel images play well with add_graph
    # tb.add_graph(model)
    # print("added model graph to tb")
    # tb.close()

    model.to(device)
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)
    else:
        pass
        # checkpoint = torch.load(args.save_model)
        # model.load_state_dict(checkpoint['state_dict'])

    if args.test:
        print("Testing")
        test(args, model, test_dataloader, device=device)

    # get features from final model.
    if args.save_features:
        loader_dict = {
            "train": train_dataloader, "test": test_dataloader, "val" : val_dataloader
        }
        get_deep_features(args, model, loader_dict, device=device)
    
