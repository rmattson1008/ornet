
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
import pickle

from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline, RandomHorizontalFlip, RandomVerticalFlip, ColorJiggle

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip(p=0.5),
            # RandomChannelShuffle(p=0.75),
            RandomThinPlateSpline(p=0.75),
        )

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) :
        print("transforming image")
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class CNN_Module(pl.LightningModule):
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model"):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.number_of_frames = number_of_frames
        self.num_classes= num_classes
        self.out_height = 128
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.label = label
        self.transform = DataAugmentation()
        # self.val_bin_accuracy = torchmetrics.BinaryAccuracy()

        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay

        # self.save_hyperparameters()
        # self.device = device

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # print(resnet)
        resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # resnet.fc = torch.nn.Linear(2048, self.out_height,bias=.6)
        resnet.fc = torch.nn.Linear(512, self.out_height)
        # resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))

        # vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
        # vgg.classifier = torch.nn.Linear(25088, self.out_height)
        # # print(vgg)

        self.cnn_layers = resnet
        # self.cnn_layers = vgg
        self.flattened_frames_size = self.number_of_frames * 1 * self.out_height
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, 10), torch.nn.Linear(10, self.num_classes) )
        # self.linear_layers.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        # self.alt_linear_layer
 

    def forward(self, x):
        # I want to classify like 3-5 frames
        frames = torch.empty((x.size(1), x.size(0) ,self.out_height)).type_as(x)

        for t in range(self.number_of_frames):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
            frames[t] = self.cnn_layers(x[:, t, :, :].unsqueeze(1))
        # frames = self.cnn_layers(x)

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



    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print(inputs.shape)
        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)

        loss = F.cross_entropy(outputs, labels)
        # return {"loss": loss}
        
        return {'loss': loss, 'pred': preds, 'target': labels}
        # Yeah so it knows the loss?? ? idk. anyway 

    def training_step_end(self, outputs):
        self.train_accuracy(outputs['pred'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_loss', outputs['loss'], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        
        return {'loss':loss, 'pred': pred, 'target': y}
        # return pred

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['pred'], outputs['target'])
        self.log('val_accuracy', self.val_accuracy)
        self.log('val_loss', outputs['loss'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        self.test_accuracy(pred, y)
        return {'loss':loss, 'pred': pred, 'target': y, 'embed': y_hat}
        # return pred

    # def test_step_end(self, outputs):
    #     self.test_accuracy(outputs['pred'], outputs['target'])
        # print(type(outputs))
        # print(len(outputs))
        # print(type(outputs[]))
        # embeds = [x['embed'] for x in outputs]
        
        # print(type(embed))
        # print('test_accuracy', self.test_accuracy.compute())
        # self.log('test_accuracy', self.test_accuracy)
        # return {'embed', outputs['embed']}
    
    def test_epoch_end(self, outputs):
       
        # print(type(outputs))
        # print(len(outputs))
        # print(type(outputs[0]))
        embeds = [x['embed'].cpu().to('cpu') for x in outputs] 
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        preds = [x['pred'].cpu().to('cpu') for x in outputs] 
        # print(type(embeds))
        # print(type(embeds[0]))
        # print(type(outputs['embed'].shape))
        # e_1 = outputs['embed']
        # e_2 = outputs['embed']
        # embeds = torch.cat((e_1, e_2))
        # list of steps(list of batch parts(array of embeddings))
        # embeds = outputs
        with open("embeddings/" + self.label +"_embeddings.pkl", "wb") as fp:   #Pickling
            pickle.dump(embeds, fp)
        with open("embeddings/" + self.label +"_targets.pkl", "wb") as fp:   #Pickling
            pickle.dump(targets, fp)
        with open("embeddings/" + self.label +"_preds.pkl", "wb") as fp:   #Pickling
            pickle.dump(preds, fp)
        # self.test_accuracy(outputs['pred'], outputs['target'])
        print('test_accuracy', self.test_accuracy.compute())
        return



class CnnLSTM_Module(pl.LightningModule):
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model"):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.number_of_frames = number_of_frames
        self.num_classes= num_classes
        self.out_height = 128
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.label = label
        self.transform = DataAugmentation()
        self.hidden_layer_size = 128
        # self.val_bin_accuracy = torchmetrics.BinaryAccuracy()

        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay

        # self.save_hyperparameters()
        # self.device = device

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # print(resnet)
        resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # resnet.fc = torch.nn.Linear(2048, self.out_height,bias=.6)
        resnet.fc = torch.nn.Linear(512, self.out_height)
        # resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))


        self.cnn_layers = resnet
        self.lstm = nn.LSTM(self.out_height,self.hidden_layer_size, 1)

        # self.flattened_frames_size = self.number_of_frames * 1 * self.out_height.... what is out size?
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_layer_size * self.number_of_frames, 10), torch.nn.Linear(10, self.num_classes) )
       

    def forward(self, x):
        # I want to classify like 3-5 frames
        frames = torch.empty((x.size(1), x.size(0) , self.out_height)).type_as(x)

        for t in range(self.number_of_frames):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
            frames[t] = self.cnn_layers(x[:, t, :, :].unsqueeze(1))

        # frames = frames.permute((1,0,2))
        # print(frames.shape)
        # flatten_frames_vector = frames.flatten(start_dim=1)
        frames = frames.flatten(start_dim=2)
        # print(frames.shape)

        #todo, switch if bi
        h0 = torch.randn(1, frames.size(1), self.hidden_layer_size).type_as(x)
        c0 = torch.randn(1,frames.size(1), self.hidden_layer_size).type_as(x)
        out, (hn, cn) = self.lstm(frames)
        out = out.permute((1,0,2))
        # print(out.shape)
        #could pool or do some sort of transform thing
        out = out.flatten(start_dim=1)
        # print(out.shape)

        logits = self.linear_layers(out)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        #TODO- use my args
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print(inputs.shape)
        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)

        loss = F.cross_entropy(outputs, labels)
        
        return {'loss': loss, 'pred': preds, 'target': labels}


    def training_step_end(self, outputs):
        self.train_accuracy(outputs['pred'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_loss', outputs['loss'], sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        
        return {'loss':loss, 'pred': pred, 'target': y}
        # return pred

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['pred'], outputs['target'])
        self.log('val_accuracy', self.val_accuracy)
        self.log('val_loss', outputs['loss'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        self.test_accuracy(pred, y)
        return {'loss':loss, 'pred': pred, 'target': y, 'embed': y_hat}
        # return pred

    # def test_step_end(self, outputs):
    #     self.test_accuracy(outputs['pred'], outputs['target'])
        # print(type(outputs))
        # print(len(outputs))
        # print(type(outputs[]))
        # embeds = [x['embed'] for x in outputs]
        
        # print(type(embed))
        # print('test_accuracy', self.test_accuracy.compute())
        # self.log('test_accuracy', self.test_accuracy)
        # return {'embed', outputs['embed']}
    
    def test_epoch_end(self, outputs):
       
        # print(type(outputs))
        # print(len(outputs))
        # print(type(outputs[0]))
        embeds = [x['embed'].cpu().to('cpu') for x in outputs] 
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        preds = [x['pred'].cpu().to('cpu') for x in outputs] 
        # print(type(embeds))
        # print(type(embeds[0]))
        # print(type(outputs['embed'].shape))
        # e_1 = outputs['embed']
        # e_2 = outputs['embed']
        # embeds = torch.cat((e_1, e_2))
        # list of steps(list of batch parts(array of embeddings))
        # embeds = outputs
        print("saving embeddings")
        with open("embeddings/" + self.label +"_embeddings.pkl", "wb") as fp:   #Pickling
            pickle.dump(embeds, fp)
        with open("embeddings/" + self.label +"_targets.pkl", "wb") as fp:   #Pickling
            pickle.dump(targets, fp)
        with open("embeddings/" + self.label +"_preds.pkl", "wb") as fp:   #Pickling
            pickle.dump(preds, fp)
        # self.test_accuracy(outputs['pred'], outputs['target'])
        print('test_accuracy', self.test_accuracy.compute())
        return


