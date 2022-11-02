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

    train_dataloader = DataLoader(train_dataset,sampler=None, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset,sampler=None, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset,sampler=None, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader


if __name__ == "__main__":
    args, _ = make_parser()
    torch.cuda.empty_cache()

    hyper_parameters = dict(
        lr=[ 0.000001],
        # lr=[.00001, .000001], 
        # lr=[0.00001], 
        # lr=[.001], 
        # lr=[.000001], 
        # lr = [0.0001 , 0.00001 ],
        # lr = [0.0001],
        # batch_size=[16, 32, 64, 91],
        # batch_size=[91, 64, 32, 16],
        # batch_size=[64, 16],
        batch_size=[32],
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
        wd =  [0, 0.1, 0.01],
        # shuffle=[True, False]
        shuffle=[False]
    )


    param_values = [v for v in hyper_parameters.values()]

    for lr,batch_size,time_steps, wd, shuffle in product(*param_values):
        args.lr = lr
        args.batch_size = batch_size
        args.shuffle = shuffle
        args.weight_decay = wd

        time_steps=5
        step=1

        comment = f' batch_size = {args.batch_size} lr = {args.lr} wd = {args.weight_decay} frames={time_steps} steps={step} lstm-kornia'
        model = CnnLSTM_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label=comment)
        # model = CNN_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label= "e-5")
        logger = TensorBoardLogger("tb_logs", name=comment)

        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True,callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=6,stopping_threshold=.02, divergence_threshold=2)])
        trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True, enable_checkpointing=False)
        # trainer = pl.Trainer(accelerator="gpu", devices=2, max_epochs=args.epochs, logger=logger, log_every_n_steps=10, strategy = "ddp_find_unused_parameters_false", deterministic=True)
        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
            args,frames_per_chunk=time_steps, step=step, resize=224)
        
        # path = '/home/rachel/ornet/tb_logs/ batch_size = 16 lr = 1e-06 wd = 0.01 frames=5 resnet18-kornia/version_3/checkpoints/epoch=56-step=1083.ckpt'
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer = pl.Trainer(accelerator="gpu", devices=1, num_nodes=1)

        trainer.test(model=model, dataloaders=test_dataloader)
        # trainer.test(model=model, ckpt_path=path, dataloaders=test_dataloader)
      
        print("leaving code loop")
        # break
print("leaving program")
# exit()
        # trainer.test(model, test_dataloader)
      

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


# nvidia-smi
# kill -9 pid
# tmux kill-session
# tmux attach

#  you need to make sure that the true data is right.
# not going to be able to get help until you clean this up and commit. 
# well... need val loss by epoch? 
# and need to name versions of the models.... 
# atleast reproduce the output. 
# have it in the state of discussion by tomorrow. \

# so for augmentation I dont think you expand it i think you use the random +/- transforms to your batch....
# go and check up on your normalizations!!!!!! oh no! pretrained inputs????
#  I really thin you are going the wrong way girly. unless val is magically wrong. 

# VISUALIZE EMBEDDINGS -> ok you got it for 3...
    # how do I visualize images right before train.
    # just save numpy... and plot in anoter location. 
# am I thinking seriously enough about the embeddings??? I think I want to see the cnn embeddings...
# MOVE TO GITHUB!!!
# RED0 AUGMENTATIONS
    # and make sure random numbers r good


# is it saving a fecking checkpoint every time??? store them on data...
# I think my model checkpooints are in there somewhere
# I should use their checkpooint process just put it on data. 
# can I overfit a batch??? Yes. And val is uch better... 
# I want to see what type of videos are messing it up


# I havent even trimmed LLO! isnt that going to confound it???
# Idk I won't have even classes then 



#?what is filling up in tb logs?
# with this I think its too small... 

# I am feeling good about
# print out what is acutally in there...


# waht do I want to have training tonight? 
# same thing w some regularization... 
# I forget what hapens with small batch size? 
# goddamn now I dont know if it was 4 or 16...


# look into normalizing
# are my augmentations doing what I think?
# get an ok one and move on to another embedder
