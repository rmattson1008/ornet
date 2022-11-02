import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from data_utils import get_accept_list
# from torchvisions
import argparse
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

from data_utils import FramePairDataset, RoiTransform, get_accept_list, TimeChunks
from models import BaseCNN, ResNet18, ResBlock, CNN_Encoder
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
import copy
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
from itertools import product

from sklearn.model_selection import train_test_split
from data_utils import FramePairDataset, RoiTransform, get_accept_list, TimeChunks
from models import BaseCNN, ResNet18, ResBlock, CNN_Encoder
from lightning_modules import CNN_Module, CnnLSTM_Module
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser
from lightning_driver import get_dataloaders

# i dont want to do this manually!

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.lr = 0.00001
    args.batch_size = 16
    args.shuffle = False
    args.weight_decay = 0.01
    args.input_dir = '/data/ornet/rachel_baselines'
    time_steps=1
    step=1
    path = '/home/rachel/ornet/tb_logs/ batch_size = 16 lr = 1e-05 wd = 0.01 frames=1 steps=1 lstm-kornia/version_0/checkpoints/epoch=4-step=1460.ckpt'

    model = CnnLSTM_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label='best-Nov2')
    logger = TensorBoardLogger("tb_logs", name="test")
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
                args,frames_per_chunk=time_steps, step=step, resize=224)
    # model = CNN_Module(number_of_frames=time_steps, num_classes =2, learning_rate= args.lr, weight_decay=args.weight_decay, label= "e-5")
    # trainer.test(model=model, ckpt_path=path, dataloaders=val_dataloader)
    # trainer.test(model=model, dataloaders=test_dataloader)

    trainer = pl.Trainer(accelerator="gpu", devices=1, num_nodes=1)
    trainer.test(model=model, ckpt_path=path, dataloaders=test_dataloader)