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
from itertools import product


def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    # lr = args.lr
    epochs = args.epochs
    print("using ADAM")
    optimizer = Adam(model.parameters(), lr=args.lr)
    # optimizer = SGD(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()
    comment = f' batch_size = {args.batch_size} lr = {args.lr} shuffle = {args.shuffle} epochs = {args.epochs} test2'
    tb = SummaryWriter(comment=comment)

    # train_losses = []
    val_losses = []

    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()

        train_loss = 0.0
        val_loss = 0.0
        total_correct = 0.0
        val_total_correct = 0.0
        train_acc = 0.0
        val_acc = 0.0
        num_batches_used = 0.0 

        for batch_idx, data in enumerate(train_dataloader):
            inputs, labels = get_augmented_batch(data)
            inputs, labels = inputs.to(device), labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            real_total_correct = outputs.argmax(dim=1).eq(labels).sum().item()
            train_acc += real_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        print(num_batches_used)
        train_loss = train_loss / num_batches_used
        train_acc = train_acc / num_batches_used * 100

        model.eval()
        num_batches_used = 0.0 
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds, _ = model(inputs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            val_total_correct += preds.argmax(dim=1).eq(labels).sum().item()
            real_val_total_correct = preds.argmax(dim=1).eq(labels).sum().item()
            val_acc += real_val_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        val_loss = val_loss / num_batches_used
        val_acc = val_acc / num_batches_used * 100
  

        tb.add_scalar("AvgTrainLoss", train_loss, epoch)
        tb.add_scalar("AvgValLoss", val_loss, epoch)
        tb.add_scalar("TrainAccuracy", train_acc, epoch)
        tb.add_scalar("ValAccuracy", val_acc, epoch)
        tb.add_scalar("TotalCorrect", total_correct, epoch)
        tb.add_scalar("ValTotalCorrect", val_total_correct, epoch)


        # save the model at lowest val loss score and continue training
        # the val loss curve is not incredibly smooth so I dont want to risk a local optimum
        val_losses.append(val_loss)
        if val_loss == np.min(val_losses):
            best_accuracy = val_acc
            if args.save_model:
                save_path = args.save_model + "_" + str(args.lr) + "_" +str(args.batch_size) + "_" +str(args.shuffle) + "_" +str(args.epochs)  + ".pth"
                # print("Saving model")
                torch.save(
                    {'epoch': epoch + 1,
                     'state_dict': model.state_dict()},
                    save_path)

    tb.add_hparams(
        {"lr": args.lr, "bsize": args.batch_size, "shuffle": args.shuffle},
        {
            # these are score at the end of training, probably overfit.
            "accuracy": train_acc,
            "loss": train_loss,
            # this is from early stop spot
            "best_accuracy": best_accuracy,
        },
    )

    tb.close()
    return


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
        print(cm) 

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
    """
    same transformation instance applied to both channels
    
    """
    inputs, labels = data
    new_images = inputs.clone().detach()
    new_labels = labels.clone().detach()
    A_transforms = [
        # [A.Sharpen(alpha=(.5, 1.), lightness=(0.1, 0.1), always_apply=True), ToTensorV2()],
        [A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=True), ToTensorV2()],
        [A.Superpixels(p_replace=0.1, n_segments=200, max_size=128, interpolation=1, always_apply=True), ToTensorV2()],
        # [A.Transpose(p=1), ToTensorV2()],
        [A.Blur(blur_limit=7, always_apply=True), ToTensorV2()],
        [A.RandomRotate90(p=1.0), ToTensorV2()],
        # [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=1.0), ToTensorV2()]
    ]

    # random.seed(73)
    for t in A_transforms:
        t = A.Compose(t, additional_targets={'image1': 'image'})
        # extend labels by one original batch
        new_labels = torch.cat((new_labels, labels))

        for sample in inputs:
            transformed = t(image=sample[0].numpy(), image1=sample[1].numpy())
            aug = torch.stack((transformed["image"], transformed["image1"]), dim=1)
            new_images = torch.cat((new_images, aug))

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
        inputs.size(0) + inputs.size(0)

    return new_images, new_labels


def get_dataloaders(args, accept_list, resize=224):
    """
    Access ornet dataset, pass any initial transformations to dataset,
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
    dataset = FramePairDataset(
        args.input_dir, accept_list=accept_list, class_types=args.classes, transform=transform)

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
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,  shuffle=args.shuffle)

    if args.weighted_samples:
        print("using weighted sampling")
        sampler = get_weighted_sampler(train_dataset, dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,  shuffle=args.shuffle)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

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

    # TODO
    if device == 'cuda':
        torch.cuda.manual_seed_all(73)
    else:
        torch.manual_seed(73)

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

    # hyper_parameters = dict(
    #     # lr=[ 0.0001, 0.00005],
    #     lr=[ 0.00005],
    #     batch_size=[16,32, 64],
    #     # shuffle=[True, False]
    #     shuffle=[True]
    # )
    # param_values = [v for v in hyper_parameters.values()]

    # for lr,batch_size, shuffle in product(*param_values):

    #     print(lr, batch_size, shuffle)
    #     args.lr = lr
    #     args.batch_size = batch_size
    #     args.shuffle = shuffle

    #     train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
    #         args, accept_list, resize=212)
    #     # model = BaseCNN()
    #     # model = VGG_Model()
    #     model = ResNet18(in_channels=2, resblock=ResBlock, outputs=3)
    #     model.to(device)
    #     print("Training")
    #     train(args, model, train_dataloader, val_dataloader, device=device)

    # model = BaseCNN()
    # # model = VGG_Model()
    model = ResNet18(in_channels=2, resblock=ResBlock, outputs=3)
    model.to(device)
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        args, accept_list, resize=212)

    if args.train:
        print("Training")
        # checkpoint = torch.load(args.save_model)
        # model.load_state_dict(checkpoint['state_dict'])
        train(args, model, train_dataloader, val_dataloader, device=device)

    if args.test:
        checkpoint = torch.load(args.save_model)
        model.load_state_dict(checkpoint['state_dict'])
        print("Testing")

        test(args, model, test_dataloader, device=device)

    # get features from final model.
    if args.save_features:
        checkpoint = torch.load(args.save_model)
        model.load_state_dict(checkpoint['state_dict'])
        print("Getting deep features")
        loader_dict = {
            "train": train_dataloader, "test": test_dataloader, "val": val_dataloader
        }
        get_deep_features(args, model, loader_dict, device=device)
