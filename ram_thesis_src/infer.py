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


from lightning_modules import CNN_Module
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import umap


from lightning_driver import get_dataloaders # I don't think this is a module or anything and exit() from this file can affect infer.py

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])



if __name__ == "__main__":

    print("inference")
    print("DID YOU DISABLE ANY STOCHASTIC PROCESSES?")
    #there must be a better way
    # why am I doing it like this. 
    # batch = ram_thesis_experiments/checkpointTEST batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 wp= 0 frames=5 steps=1 dropout=True cnn-squeeze-lstm-16.json
    # label = f'batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 wp= 0 frames=5 steps=1 dropout=False cnn-squeeze-lstm-16'
    label = f'batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 wp= 0 frames=5 steps=1 dropout=True cnn-squeeze-lstm-16'
    # label = "TEST batch_size = 16 shuffle=False lr = 1e-05 wd = 0.0 wp= 0 frames=5 steps=1 dropout=False cnn-squeeze-lstm'
    with open(f"ram_thesis_experiments/checkpoint{label}.json", "r") as f:
        model_card = json.load(f)
    args = Dict2Class(model_card['args'])

    print(args.agg)
    print()

    model = CNN_Module(number_of_frames= args.time_steps, num_classes=2, learning_rate= args.lr, weight_decay=args.weight_decay, label= args.comment,  dropout=False, aggregator=args.agg)
    print(model)
    print(model.state_dict().keys())
    print(model_card['checkpoint'])

    # for some reason you have to do this the pytorch way
    # linear layers of the agg have trouble getting the weight and bias to transfer.
    # maybe they are not corrrectly added to the pl module?
    checkpoint= torch.load(model_card['checkpoint'])
    print("loaded checkpoint")
    # print(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # model = model.load_from_checkpoint(checkpoint_path=model_card['checkpoint'])
    # print(model)
    logger = TensorBoardLogger("tb_logs", name=args.comment)

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
                args,frames_per_chunk=args.time_steps, resize=224)

    print("Loaded test hparams")

    trainer = pl.Trainer(accelerator='gpu',devices=1, num_nodes=1, logger=logger)

    print("loaded trainer")
    trainer.test(model=model, dataloaders=test_dataloader)

    




    exit()



    # TODO - pick better dim reduciton that gmm will find two components... 



