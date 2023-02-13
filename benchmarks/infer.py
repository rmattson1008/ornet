import os
from configargparse import ArgParser
import json
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


from data_utils import  get_accept_list, TimeChunks
from lightning_modules import CNN_Module
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser
from sklearn.model_selection import train_test_split

# from lightning_driver import get_dataloaders

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_dataloaders(args,time_steps=3, frames_per_chunk=3, resize=224):
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
    train_dataset = TimeChunks(args.input_dir, accept_list=X_train, frames_per_chunk=frames_per_chunk, step=args.step, transform=transform)  
    print("creating val dataset")
    val_dataset = TimeChunks(args.input_dir, accept_list=X_val, frames_per_chunk=frames_per_chunk,step=args.step,   transform=transform)  
    
    print("creating test dataset")
    test_dataset = TimeChunks(args.input_dir, accept_list=X_test, frames_per_chunk=frames_per_chunk, step=args.step,  transform=transform, verbose=True, shuffle_chunks=False)  

    train_dataloader = DataLoader(train_dataset,shuffle=args.shuffle, sampler=None, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset,shuffle=args.shuffle,sampler=None, batch_size=1, num_workers=4)
    val_dataloader = DataLoader(val_dataset,shuffle=args.shuffle,sampler=None, batch_size=args.batch_size, num_workers=4)

    return train_dataloader, test_dataloader, val_dataloader




if __name__ == "__main__":

    #there must be a better way
    with open("checkpoint.json", "r") as f:
        model_card = json.load(f)
    args = Dict2Class(model_card['args'])

    model = CNN_Module(number_of_frames= args.time_steps, num_classes=2, learning_rate= args.lr, weight_decay=args.weight_decay, label= args.comment,  dropout=args.dropout)
    model = model.load_from_checkpoint(checkpoint_path=model_card['checkpoint'])
    logger = TensorBoardLogger("tb_logs", name=args.comment)

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
                args,frames_per_chunk=args.time_steps, resize=224)

    print("Loaded test hparams")

    trainer = pl.Trainer(accelerator='gpu',devices=1, num_nodes=1, logger=logger)
    trainer.test(model=model, dataloaders=test_dataloader)

    exit()

