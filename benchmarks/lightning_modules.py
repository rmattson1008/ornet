
from typing import OrderedDict
import torch
from collections import OrderedDict 
import torch.nn as nn
import numpy as np
from torch.nn import Linear, ReLU, ELU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d, LeakyReLU
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer









class CNN_Module(pl.LightningModule):
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.number_of_frames = number_of_frames
        self.num_classes= num_classes
        self.out_height = 16
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.lr = learning_rate
        self.wd = weight_decay



        # self.save_hyperparameters()
        # self.device = device

        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        resnet18.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18.fc = torch.nn.Linear(512, self.out_height)

        self.cnn_layers = resnet18
        self.flattened_frames_size = self.number_of_frames * 1 * self.out_height
        self.linear_layers = torch.nn.Linear(self.flattened_frames_size, self.num_classes)
 

    def forward(self, x):
        # I want to classify like 3-5 frames
        frames = torch.empty((x.size(1), x.size(0) ,self.out_height)).type_as(x)

        for t in range(self.number_of_frames):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
            frames[t] = self.cnn_layers(x[:, t, :, :].unsqueeze(1))

        frames = frames.permute((1,0,2))
        flatten_frames_vector = frames.flatten(start_dim=1)
        # print("flat shape" ,flatten_frames_vector.shape)

        # print("")
        logits = self.linear_layers(flatten_frames_vector)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        #TODO- use my args
        return optimizer

    def training_step_end(self, outputs):
        # update and log
        self.train_accuracy(outputs['preds'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        self.log('train_loss', outputs['loss'], on_step=True,  on_epoch=False, sync_dist=True)
        return
    
    def training_epoch_end(self, outputs):
        self.log('train_accuracy_ep', self.train_accuracy,on_step=False, on_epoch=True)
        # self.log('train_loss', outputs['loss'], on_epoch=True, sync_dist=True)
        return
    

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, labels = batch
        # print("train labels", labels)
            
            # zero the parameter gradients
        # optimizer.zero_grad()

        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)

        loss = F.cross_entropy(outputs, labels)
        # return {"loss": loss}
        
        return {'loss': loss, 'preds': preds, 'target': labels}
    
    # t >= 0 && t < n_classes

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        preds = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        # print("val labels", y)
        # Immediately driving everything towards class 1
        # y_pred = self(x)       
        return {'val_loss': val_loss, 'preds': preds, 'target': y}

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['preds'], outputs['target'])
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=False)
        self.log('val_loss', outputs['val_loss'], on_step=True, on_epoch=False, sync_dist=True)
        return

    def validation_epoch_end(self, outputs):
        self.log('val_accuracy_ep', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_loss_ep', outputs['val_loss'],on_step=False, on_epoch=True)
        return
    

    def test_step(self, batch, batch_idx):
        x, labels= batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, labels)
        preds = y_hat.softmax(dim=-1)
        print("preds" , preds)
        print("labels", labels)
        # acc = self.accuracy(preds, )
        return {'test_loss': loss, 'preds': preds, 'target': labels}

    # def test_step_end(self, outputs):


    # def training_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     # predictions = batch_parts["pred"]
    #     # losses from each GPU
    #     losses = batch_parts["loss"]

    #     # gpu_0_prediction = predictions[0]
    #     # gpu_1_prediction = predictions[1]

    #     # do something with both outputs
    #     return (losses[0] + losses[1]) / 2

