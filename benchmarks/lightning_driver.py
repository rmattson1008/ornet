import os
from platform import node
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy # set ddps flag
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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

from data_utils import  get_accept_list, TimeChunks
from lightning_modules import CNN_Module, CnnLSTM_Module
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from time import sleep

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torch.utils.tensorboard import SummaryWriter
from itertools import product

from sklearn.model_selection import train_test_split


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # print(data.shape)
        # Mean over batch, height and width, but not over the channels
        # channels_sum += torch.mean(data, dim=[0,1,3,4])
        # channels_squared_sum += torch.mean(data**2, dim=[0,1,3,4])
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_dataloaders(args,time_steps=3, frames_per_chunk=3, step =1, resize=224):
    """
    Access ornet dataset, pass any initial transformations to dataset,
    split into train/test/validate, and return dataloaders
    """
    X, y = get_accept_list("/data/ornet/gmm_intermediates", ['control', 'mdivi', 'llo'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=39)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=resize)])
    # TODO - some sort of normalizing step?
    # transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize(size=224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = TimeChunks(args.input_dir, accept_list=X_train, frames_per_chunk=frames_per_chunk, step=step, transform=transform)  
    test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=frames_per_chunk, step=step,  transform=transform)  
    val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=frames_per_chunk,step=step,   transform=transform)  

    train_dataloader = DataLoader(train_dataset,sampler=None, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset,sampler=None, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset,sampler=None, batch_size=args.batch_size, num_workers=4)

    return train_dataloader, test_dataloader, val_dataloader


if __name__ == "__main__":
    args, _ = make_parser()
    torch.cuda.empty_cache()
    pl.seed_everything(42, workers=True)

    hyper_parameters = dict(
        lr=[ 0.00001],
        # lr=[.00001, .000001], 
        # lr=[0.00001], 
        # lr=[.001], 
        # lr=[.000001], 
        # lr = [0.0001 , 0.00001 ],
        # lr = [0.0001],
        # batch_size=[16, 32, 64, 91],
        # batch_size=[91, 64, 32, 16],
        # batch_size=[64, 16],
        batch_size=[16],
        time_steps=[5],
        # batch_size=[ ],
        # batch_size=[91],
        # batch_size=[ 32],
        # batch_size=[1],
        # batch_size=[32],
        # roi=[True, False],
        # roi=[False],
        # wd = [0.2, 0.1, 0],
        # wd = [.1],
        wd =  [0.0],
        # shuffle=[True, False]
        shuffle=[False, True]
    )

    # idk what im doing!!!
    param_values = [v for v in hyper_parameters.values()]

    for lr,batch_size,time_steps, wd, shuffle in product(*param_values):
        args.lr = lr
        args.batch_size = batch_size
        args.shuffle = shuffle
        args.weight_decay = wd

        time_steps=time_steps
        step=1


        comment = f' batch_size = {args.batch_size} shuffle={shuffle} lr = {args.lr} wd = {args.weight_decay} frames={time_steps} steps={step} cnn-squeeze'
        # model = CnnLSTM_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label=comment, dropout=False)
        model = CNN_Module(number_of_frames=time_steps, num_classes=2, learning_rate= args.lr, weight_decay=args.weight_decay, label= comment,  dropout=False)
        logger = TensorBoardLogger("tb_logs", name=comment)

        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True,callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=6,stopping_threshold=.02, divergence_threshold=2)])
        trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True, enable_checkpointing=False)
        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True)
        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
            args,frames_per_chunk=time_steps, step=step, resize=224)

        trainer.fit(model, train_dataloader, val_dataloader)
        
        # mean, std = get_mean_and_std(train_dataloader)
        # print(mean, std)
        
        # path = '/home/rachel/ornet/tb_logs/ batch_size = 16 lr = 1e-06 wd = 0.01 frames=5 resnet18-kornia/version_3/checkpoints/epoch=56-step=1083.ckpt'
        # path = '/home/rachel/ornet/tb_logs/ batch_size = 16 lr = 1e-06 wd = 0.01 frames=5 resnet18-kornia/version_3/checkpoints/epoch=56-step=1083.ckpt'
        # trainer = pl.Trainer(accelerator="gpu", devices=1, num_nodes=1)
        # trainer.test(model=model, dataloaders=test_dataloader)
        # trainer.test(model=model, ckpt_path=path, dataloaders=test_dataloader)
      
        print("leaving code loop")
        # break
print("leaving program")
# exit()


# TODO - debug the lstm. Are all the connections fed in and out?
# TODO - use dim reductions to look at non final embedding - LOOK FARTHER BACK
# TODO - actually picture the augmentations
# TODO - trace back timestamp (check verbose) (don't rndomize test set)
    #for augmentations, can I black out just one frame?
# DONE - See if further normalization should be used 

# RegNet