import os
import math
from turtle import st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torchvision
from torch.optim import Adam, SGD, AdamW
from torch.nn import CrossEntropyLoss
import pickle

from data_utils import FramePairDataset, RoiTransform, get_accept_list, TimeChunks, TimeChunks2
from models import BaseCNN, ResNet18, ResBlock, CNN_Encoder
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

from sklearn.model_selection import train_test_split


def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    print("device in train", device)
    print(torch.cuda.current_device())
    # lr = args.lr
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = SGD(model.parameters(), lr=args.lr)
    print(optimizer)
    criterion = CrossEntropyLoss()
    mn = args.save_model.split("/")[-1]
    comment = f' batch_size = {args.batch_size} lr = {args.lr} shuffle = {args.shuffle} roi = {args.roi} epochs = {args.epochs} wd = {args.weight_decay} name = {mn} res_adam'
    tb = SummaryWriter(comment=comment)

    # train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # TODO - print input 
        print("epoch", epoch)
    # for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()

        train_loss = 0.0
        val_loss = 0.0
        total_correct = 0.0
        val_total_correct = 0.0
        train_acc = 0.0
        val_acc = 0.0
        num_batches_used = 0.0 
        other_val_loss = 0.0

        for batch_idx, data in enumerate(tqdm(train_dataloader, desc="epoch")):
            # inputs, labels = get_augmented_batch(data)
            inputs, labels = data
            # inputs = inputs.unsqueeze(0)
            # inputs = inputs.permute(1,0,2,3)
            # inputs = inputs.unsqueeze(0)
            # print(inputs.shape)

            print("input shape", inputs.shape)
            # print("input max", torch.max(inputs[0]))
            inputs, labels = inputs.to(device), labels.to(device)
            # print("input device", inputs.get_device())
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs, _ = model(inputs)
            outputs = model(inputs)
            # is this the correct dim?
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            # no, we are not softmaxing before cross entropy loss... is this a mistake in previous? other models end with softmax
            # oh no they don't but the accuracy may be wrong... 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            real_total_correct = outputs.argmax(dim=1).eq(labels).sum().item()
            train_acc += real_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        train_loss = train_loss / num_batches_used
        train_acc = train_acc / num_batches_used * 100

        model.eval()
        num_batches_used = 0.0 
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            # preds, _ = model(inputs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            other_val_loss += loss.item() 
            val_total_correct += preds.argmax(dim=1).eq(labels).sum().item()
            real_val_total_correct = preds.argmax(dim=1).eq(labels).sum().item()
            val_acc += real_val_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        other_val_loss = other_val_loss / len(val_dataloader)
        val_loss = val_loss / num_batches_used
        val_acc = val_acc / num_batches_used * 100
  

        tb.add_scalar("AvgTrainLoss", train_loss, epoch)
        tb.add_scalar("AvgValLoss", val_loss, epoch)
        tb.add_scalar("OtherValLoss", other_val_loss, epoch)
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
                save_path = args.save_model + "_" + str(args.lr) + "_" +str(args.batch_size) + "_" +str(args.shuffle) + "_" +str(args.epochs) + "_" +str(args.weight_decay)  + ".pth"
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
        # y_true = []
        y_pred = torch.tensor([]).to(device)
        # y_pred = []
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            # calculate outputs by running images through the network
            # outputs, _ = model(images)
            outputs = model(images)
            # outputs = torch.nn.functional.softmax(outputs[0], dim=0)
            # print("out shape", output.shape)
            probabilities = torch.nn.functional.softmax(outputs, dim=0)
            # print("prob", probabilities.shape)
            # print(probabilities)

            predicted = torch.argmax(probabilities, dim=1)

            # print("labels" , labels)
            # print("predicted", predicted)
            # y_true.append(labels.cpu())
            # y_pred.append(predicted.cpu())
            # print("pred", predicted.shape)
            # _, predicted = torch.max(outputs.data, 1)

            # print("pred", predicted.shape)
            # predicted = torch.max(outputs.data)
            # bad approach?
            y_pred = torch.cat((y_pred, predicted), 0)
            y_true = torch.cat((y_true, labels), 0)
    # print(y_true)
    # print(y_pred)
    # print to file?
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())

    assert len(y_pred) == len(y_true)
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print("Accuracy:", accuracy)

    if show_plots:
        # print(args.classes)
        print(cm) 

    return


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


def get_dataloaders(args,time_steps=3, frames_per_chunk=3, step =1, resize=224):
    """
    Access ornet dataset, pass any initial transformations to dataset,
    split into train/test/validate, and return dataloaders
    """
    X, y = get_accept_list("/data/ornet/gmm_intermediates", ['control', 'mdivi', 'llo'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.10, random_state=42)
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=resize)])
    # TODO - some sort of normalizing step?
#     # transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize(size=224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # train_dataset = TimeChunks(args.input_dir, accept_list=X_train, frames_per_chunk=frames_per_chunk, step=step, transform=transform)  
    # test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=frames_per_chunk, step=step,  transform=transform)  
    # val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=frames_per_chunk,step=step,   transform=transform)  

    # train_dataset = TimeChunks2(args.input_dir, accept_list=X_train, num_chunks=time_steps, frames_per_chunk=frames_per_chunk, step=step, transform=transform)  
    # test_dataset = TimeChunks2(args.input_dir, accept_list=X_test, num_chunks=time_steps ,frames_per_chunk=frames_per_chunk, step=step,  transform=transform)  
    # val_dataset = TimeChunks2(args.input_dir, accept_list=X_val, num_chunks=time_steps ,frames_per_chunk=frames_per_chunk,step=step,   transform=transform)  


    train_dataset = TimeChunks(args.input_dir, accept_list=X_train,  frames_per_chunk=frames_per_chunk, step=step, transform=transform)  
    test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=frames_per_chunk, step=step,  transform=transform)  
    val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=frames_per_chunk, step=step,   transform=transform)  
# # sampler = RandomSampler(dataset, replacement=False, num_samples=10, generator=None)

    # import torchvision.datasets as datasets
    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # # test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_dataloader = val_dataloader

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


    return feature_dict


if __name__ == "__main__":

    args, _ = make_parser()
    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else "cuda"
    print("Device:", device)
    device = torch.device(device)
    print("Device:", device)
    # "cuda:1,3"

    print(f'available devices: {torch.cuda.device_count()}')
    # print(device)
    # TODO
    if device == 'cuda':
        torch.cuda.manual_seed_all(73)
    else:
        torch.manual_seed(73)

    # X, y = get_accept_list("/data/ornet/gmm_intermediates", ['control', 'mdivi', 'llo'])


#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#     X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.10, random_state=42)

#     step = 5
#     transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     # transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize(size=224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     train_dataset = TimeChunks(args.input_dir, accept_list=X_train, frames_per_chunk=3, step=step, transform=transform)  
#     test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=3, step=step,  transform=transform)  
#     val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=3,step=step,   transform=transform)  
# # sampler = RandomSampler(dataset, replacement=False, num_samples=10, generator=None)

#     train_dataloader = DataLoader(train_dataset, batch_size=64)
#     test_dataloader = DataLoader(test_dataset, batch_size=64)
#     val_dataloader = DataLoader(val_dataset, batch_size=64)
    # """
    # we have more segmented cell videos saved on logan then intermediates.
    # best to rerun ornet on raw data, but till discrepancy is resolved this
    # will result in using the 114 samples that Neelima used in the last scipy submit
    # class balance 29, 31, 54
    # """
    # path_to_intermediates = "/data/ornet/gmm_intermediates"
    # accept_list = []
    # for subdir in args.classes:
    #     path = os.path.join(path_to_intermediates, subdir)
    #     files = os.listdir(path)
    #     accept_list.extend([x.split(".")[0]
    #                        for x in files if 'normalized' in x])

    hyper_parameters = dict(
        # lr=[ 0.0001, 0.00001],
        lr=[0.00001], 
        # lr = [0.0001 , 0.00001 ],
        # lr = [0.0001],
        # batch_size=[16, 32, 64, 91],
        # batch_size=[91, 64, 32, 16],
        # batch_size=[64, 16],
        batch_size=[16],
        # batch_size=[91],
        # batch_size=[ 32],
        # batch_size=[1],
        # batch_size=[32],
        # roi=[True, False],
        roi=[False],
        wd = [0, 0.1, 0.01],
        shuffle=[True]
        # shuffle=[False]
    )

    #finish roi res at 1e-05

    param_values = [v for v in hyper_parameters.values()]

    for lr,batch_size,roi,wd, shuffle in product(*param_values):

        print(lr, batch_size, shuffle)
        args.lr = lr
        args.batch_size = batch_size
        args.shuffle = shuffle
        args.roi = roi
        args.weight_decay = wd

        # train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        #     args, accept_list, resize=224)

        # model = BaseCNN()
        # model = VGG_Model()   
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # print(model)
        # import torchvision.models as models
        # model = models.video.r3d_18(pretrained=True)
        time_steps = 5
        model = CNN_Encoder(number_of_frames =time_steps, device=device)
        model= nn.DataParallel(model, device_ids = [0, 1])
        print("arg", args.cuda)
        print("torch device", torch.cuda.get_device_name())
        print("device2", device)
        model.to(device)
        print("model on device", next(model.parameters()).device)
        # print(model)
        # model.fc = torch.nn.Linear(512, 10)

        # model = ResNet18(in_channels=2, resblock=ResBlock, outputs=3)
        # model.to(device)

        # print("Training")

        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
            args, time_steps=0, frames_per_chunk=time_steps, step=1, resize=224)
        train(args, model, train_dataloader, val_dataloader, device=device)

    # model = BaseCNN()
    # # model = VGG_Model()
    # model = ResNet18(in_channels=2, resblock=ResBlock, outputs=3)
    # model.to(device)

    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)

    if args.test:
        # checkpoint = torch.load(args.save_model)
        # model.load_state_dict(checkpoint['state_dict'])
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
        feature_dict = get_deep_features(args, model, loader_dict, device=device)
        # with open(args.save_features, 'wb') as f:
        #     pickle.dump(feature_dict, f)


#todo -
# change to prob dist
# try 3 classes
# try further out. (more steps)
# try 3d https://paperswithcode.com/model/resnet-3d?variant=resnet-3d-18#:~:text=ResNet%203D%20is%20a%20type,convolutions%20in%20the%20top%20layers.


# can you please switch to SGD with no weight decay? eh... adam is safe
# first make sure the model can overfit the data? 100% train acc
# dont use pretrain?
#  was the good one when you hadn't normalized? set to 1?
# tomorrow commit asap


# THE SPACED OUT 2D MODEL DIDNT WORK
# ONLY 64 has BEEN TRIED
# WHY IS INITIAL VALUE CHANGING

# pretty good results for fufi5 ... 80% accuracy

# Anyway think about embedding space... 

#batch size 1 doesnt work at all for time chunks... go back to data abandon this. 
# PLOT SOME RESULTS IG
# ignore anything pretrained... 
# you don't want to be messing with channels
# look back at the pooling choices... I dont think you want to pool early
# unless you acctually use 3d convulution.... and red isnt meant for you. 
