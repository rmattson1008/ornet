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
from lightning_modules import CNN_Module
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
    print("creating train dataset")
    train_dataset = TimeChunks(args.input_dir, accept_list=X_train, frames_per_chunk=frames_per_chunk, step=step, transform=transform)  
    print("creating val dataset")
    val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=frames_per_chunk,step=step,   transform=transform)  
    
    print("creating test dataset")
    test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=frames_per_chunk, step=step,  transform=transform, verbose=True, shuffle_chunks=False)  

    train_dataloader = DataLoader(train_dataset,shuffle=args.shuffle, sampler=None, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset,shuffle=args.shuffle,sampler=None, batch_size=1, num_workers=4)
    val_dataloader = DataLoader(val_dataset,shuffle=args.shuffle,sampler=None, batch_size=args.batch_size, num_workers=4)

    return train_dataloader, test_dataloader, val_dataloader

def trials():
    return

if __name__ == "__main__":
    args, _ = make_parser()
    torch.cuda.empty_cache()
    pl.seed_everything(42, workers=True)


    # batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 frames=5 steps=1 lstm-squeeze/version_4
    # batch_size = 16 shuffle=True lr = 1e-05 wd = 0.0 frames=5 steps=1 lstm-squeeze/version_1


    # batch_size = 16 shuffle=True lr = 1e-05 wd = 0.0 frames=5 steps=1 cnn-squeeze/version_0
    # batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 frames=5 steps=1 cnn-squeeze/version_0

    # for more epochs?
   ## batch_size = 32 shuffle=False lr = 1e-05 wd = 0.0 frames=5 steps=1 cnn-squeeze/version_0
    # batch_size = 32 shuffle=True lr = 1e-05 wd = 0.0 frames=5 steps=1 cnn-squeeze/version_0

    hyper_parameters = dict(
        lr=[0.00001],
        batch_size=[16],
        time_steps=[5],
        wd =  [0.0],
        dropout = [True],
        shuffle=[False]
    )

    # idk what im doing!!!
    param_values = [v for v in hyper_parameters.values()]

    for lr, batch_size, time_steps, wd, dropout, shuffle in product(*param_values):
        args.lr = lr
        args.batch_size = batch_size
        args.shuffle = shuffle
        #TODO I dont think I'm using shuffle... 
        args.weight_decay = wd

        time_steps=time_steps
        step=1
        dropout=dropout

        comment = f' batch_size = {args.batch_size} shuffle={shuffle} lr = {args.lr} wd = {args.weight_decay} frames={time_steps} steps={step} dropout={dropout} cnn-squeeze'
        # model = CnnLSTM_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label=comment, dropout=False)
        model = CNN_Module(number_of_frames=time_steps, num_classes=2, learning_rate= args.lr, weight_decay=args.weight_decay, label= comment,  dropout=dropout)
        logger = TensorBoardLogger("tb_logs", name=comment)

        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True,callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=6,stopping_threshold=.02, divergence_threshold=2)])
        trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True, enable_checkpointing=False)
        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True)
        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
            args,frames_per_chunk=time_steps, step=step, resize=224)
        print("test:", len(test_dataloader))

        trainer.fit(model, train_dataloader, val_dataloader)


        # trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, deterministic=True, enable_checkpointing=False)
        # path = "/home/rachel/ornet/tb_logs/ batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 frames=5 steps=1 cnn-squeeze/version_97/checkpoints/epoch=49-step=2900.ckpt"
        # model = model.load_from_checkpoint(path)
        trainer.test(model=model, dataloaders=test_dataloader)
        # trainer.test(model=model ,dataloaders=test_dataloader)

    
    #   If you want to stop a training run early, you can press “Ctrl + C” on your keyboard. The trainer will catch the KeyboardInterrupt and attempt a graceful shutdown, including running accelerator callback on_train_end to clean up memory. The trainer object will also set an attribute interrupted to True in such cases. If you have a callback which shuts down compute resources, for example, you can conditionally run the shutdown logic for only uninterrupted runs.
        print("leaving code loop")
        # break
print("leaving program")
exit()




# TODO - debug the lstm. Are all the connections fed in and out?
# TODO - use dim reductions to look at non final embedding - LOOK FARTHER BACK!!!
# DONE - actually picture the augmentations 
# TODO - trace back timestamp (check verbose) (don't randomize test set)
    #for augmentations, can I black out just one frame?
# DONE - See if further normalization should be used 
# Truly i keep forgetting which hyperparams are good\
#todo - ju



# TODO - clear up calling test error
# finish hooking Zs
# get umap of test and train
# What happens with 3 classes? 

#Regnet

