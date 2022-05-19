import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
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


def train(args, train_dataloader, val_dataloader, device='cpu'):
    lr = args.lr
    epochs = args.epochs
    # optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
    # for epoch in tqdm(range(epochs), desc ="epochs"):
        model.train()
        training_loss = 0.0
        for data in train_dataloader:
            inputs, labels = get_augmented_batch(data)

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()  # shouldn't stay on this step.
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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
            pred, _ = model(inputs)
            loss = criterion(pred, labels)
            valid_loss += loss.item()


        print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
                \t\t Validation Loss: {valid_loss / len(val_dataloader )}')

        val_loss_epoch = valid_loss / len(val_dataloader)
        train_loss_epoch = training_loss / len(train_dataloader)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        
        # print(np.min(val_losses))
        if val_loss_epoch == np.min(val_losses):
            print("Saving model")
            torch.save(
                {'epoch': epoch + 1,
                'state_dict': model.state_dict()},
                args.save_model)

    plot_dict={"train_loss": train_losses, "val_loss": val_losses}

    return plot_dict
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


def test(args, test_dataloader, show_plots=True, device='cpu'):
    model.eval()  # is necessary?


    with torch.no_grad():
        y_true=torch.tensor([]).to(device)
        y_pred=torch.tensor([]).to(device)
        for images, labels in test_dataloader:
            images, labels=images.to(device), labels.to(device)
            images=images.float()
            # calculate outputs by running images through the network
            outputs, _=model(images)
            _, predicted=torch.max(outputs.data, 1)
            # bad approach?
            y_pred=torch.cat((y_pred, predicted), 0)
            y_true=torch.cat((y_true, labels), 0)

    cm=confusion_matrix(y_true.cpu(), y_pred.cpu())

    assert len(y_pred) == len(y_true)
    accuracy=(y_true == y_pred).sum() / len(y_true)
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
    indices=train_dataset.indices
    y_train=[dataset.targets[i] for i in indices]  # Messed up this
    train_dataset_count=np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights=1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights=weights[y_train]

    sampler=WeightedRandomSampler(weights, len(train_dataset))

    return sampler


def get_augmented_batch(data):
    inputs, labels=data
    new_images=inputs.clone().detach()
    new_labels=labels.clone().detach()
    A_transforms=[
        [A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1), ToTensorV2()],
        [A.Transpose(p=1), ToTensorV2()],
        [A.Blur(blur_limit=7, always_apply=True), ToTensorV2()]
        # [A.RandomShadow((0,0,1,1), 5, 10, 3, p=1), ToTensorV2()],
    ]

    for t in A_transforms:
        t=A.Compose(t)
        # extend labels by one original batch
        new_labels=torch.cat((new_labels, labels))

        for sample in inputs:
            aug=[t(image=channel.numpy())["image"]
                   for channel in sample]
            aug=torch.stack(aug, dim=1)
            new_images=torch.cat((new_images, aug))

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
                           inputs.size(0) + inputs.size(0)

    return new_images, new_labels


def get_dataloaders(args, accept_list, resize=28):
    """
    Access ornet dataset, apply any necessary transformations to images,
    split into train/test/validate, and return dataloaders
    """

    if args.roi:
        print("Using ROI inputs")
        # default interpolation is bilinear, no idea if there is better choice
        transform=transforms.Compose(
            [RoiTransform(window_size=(28, 28)), transforms.Resize(size=resize)])
    else:
        print("Using global image inputs")
        transform=transforms.Compose([transforms.Resize(size=resize)])
        # print("!!!!! using 224x224")
        # transform = transforms.Compose([transforms.Resize(size=224)])
    # TODO - normalize??
    # dataset = FramePairDataset(args.input_dir, class_types=args.classes)
    dataset=FramePairDataset(
        args.input_dir, accept_list=accept_list, class_types=args.classes, transform=transform)
    # dataset = augment_dataset(dataset, transform)

    assert len(dataset) > 0

    # don't think this is the best way to do this... needs even percent split to work
    # brittle
    train_split=.8
    train_size=math.floor(train_split * len(dataset))
    test_size=math.ceil((len(dataset) - train_size) / 2)
    val_size=math.floor((len(dataset) - train_size) / 2)
    train_size, test_size, val_size=int(
        train_size), int(test_size), int(val_size)

    assert train_size + val_size + test_size == len(dataset)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed())
    train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(73))
    test_dataloader=DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size)

    # augment images
    # print(len(train_dataset))
    # train_dataset = augment_dataset(train_dataset, transform)
    # print(len(train_dataset))

    if args.weighted_samples:
        print("using weighted sampling")
        sampler=get_weighted_sampler(train_dataset, dataset)
        train_dataloader=DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader=DataLoader(
            train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader

def get_deep_features(args, loaders=[], device="cpu"):
    print("Getting deep features")
    feature_dict={"control": [], "mdivi": [], "llo": []}
    model.eval()
    # l = np.array()
    frames=[]
    for loader in loaders:
        for inputs, labels in loader:
            inputs, labels=inputs.to(device), labels.to(device)
            inputs=inputs.float()
            outputs, features=model(inputs)

            labels=labels.cpu().detach().numpy()
            features=features['embeddings10'].cpu().detach().numpy()

            df=pd.DataFrame(features)
            df['label']=labels
            frames.append(df)
    final_df=pd.concat(frames)
    # print(final_df)

    return final_df


# def get_augmentations():

#     # datasets = []

#     # datasets.append(dataset)
#     # didnt even include the original transform...

#     albument_Ts = [
#         [A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p = 1), ToTensorV2()],
#         [A.Transpose(p=1), ToTensorV2()],
#         [A.Blur(blur_limit=7, always_apply=True), ToTensorV2()]
#         # [A.RandomShadow((0,0,1,1), 5, 10, 3, p=1), ToTensorV2()],
#     ]

#     # Can't get torch transforms and albumentations transforms to play nicely
#     torch_Ts = [
#         transforms.RandomAffine(0, shear = (20,40)),
#         transforms.RandomAffine(0, shear = (-10,10))
#         ]
#     # try some slighter transforms????

#     # for t in albument_Ts:
#     #     ds = copy.deepcopy(dataset)
#     #     ds.aug = A.Compose(t)
#     #     datasets.append(ds)

#     # for t in torch_Ts:
#     #     ds = copy.deepcopy(dataset)
#     #     ds.transform = transforms.Compose([base_transform, t])
#     #     datasets.append(ds)

#     # anyway is making deep copies the best way to do this?
#     return albument_Ts
#     # return


if __name__ == "__main__":

    args, _=make_parser()
    device='cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    device=torch.device(device)

    """
    we have more segmented cell videos saved on logan then intermediates.
    best to rerun ornet on raw data, but till discrepancy is resolved this
    will result in using the 114 samples that Neelima used in the last scipy submit
    class balance 29, 31, 54
    """
    path_to_intermediates="/data/ornet/gmm_intermediates"
    accept_list=[]
    for subdir in args.classes:
        path=os.path.join(path_to_intermediates, subdir)
        files=os.listdir(path)
        accept_list.extend([x.split(".")[0]
                           for x in files if 'normalized' in x])

    train_dataloader, test_dataloader, val_dataloader=get_dataloaders(
        args, accept_list, resize=212)

    # model = BaseCNN()
    # model = VGG_Model()
    model=ResNet18(in_channels=2, resblock=ResBlock, outputs=3)
    model.to(device)

    if args.train:
        print("Training")
        loss_dict = train(args, train_dataloader, val_dataloader, device=device)

        if args.save_losses:
            print("saving loss plots")
            path = "/home/rachel/ornet/losses/l2.pkl"
            with open(path, "wb") as tf:
                pickle.dump(loss_dict, tf)
    else:
        checkpoint=torch.load(args.save_model)
        model.load_state_dict(checkpoint['state_dict'])

    if args.test:
        print("Testing")
        test(args, test_dataloader, device=device)

    # get features from final model.
    if args.save_features:
        #make sure we on best model
        checkpoint=torch.load(args.save_model)
        model.load_state_dict(checkpoint['state_dict'])
        print("best performance at epoch", checkpoint['epoch'])

        loaders=[train_dataloader, test_dataloader, val_dataloader]
        df=get_deep_features(args, loaders, device=device)
        # TODO - test deep features method
        # save_path="/home/rachel/representations/cnn/BaseCnn_embeddings10_roi.pkl"
        save_path=args.save_features

        with open(save_path, 'wb') as f:
            pickle.dump(df, f)
