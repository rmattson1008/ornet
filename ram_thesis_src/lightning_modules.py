
from pickletools import string1
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
import umap
import umap.plot

from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline, RandomHorizontalFlip, RandomVerticalFlip, ColorJiggle
import kornia.augmentation as K
import matplotlib.pyplot as plt
import matplotlib
import logging
from sklearn.mixture import GaussianMixture as GMM
from torch.nn import LSTM


from torchmetrics.classification import Accuracy


features = {}

def get_features(name):
    def hook(model, input, output):    
        batch = input[0].detach()
        features[name] = batch
        # print("batch hook", batch.shape)
    return hook

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = K.VideoSequential(
            # K.Normalize([0.0024, 0.0024, 0.0024], [0.0149, 0.0149, 0.0150], p=1.0),
            K.Normalize([0.0024, 0.0024, 0.0024, 0.0024, 0.0024], [0.0149, 0.0149, 0.0150, 0.0150, 0.0150], p=1.0),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),

        )
        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) :
        # print("transforming image")
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class CNN_Module(pl.LightningModule):
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model", dropout=False, aggregator='lstm'):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.number_of_frames = number_of_frames
        self.num_classes= num_classes
        self.out_height = 8
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.gmm_accuracy = torchmetrics.Accuracy(task='binary')
        self.annotation = label
        self.transform = DataAugmentation()
        self.dropout = dropout
        self.show_image = False
        self.gmm = None


        assert aggregator in ['flatten', 'lstm', 'mean']
        self.aggregator_type = aggregator
        
    
        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay
        
        squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
        squeeze.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.cnn_layers = squeeze
        self.fc = torch.nn.Linear(1000, self.out_height)

        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.out_height, self.num_classes))
        self.flattened_frames_size = self.number_of_frames * 1 * self.out_height # wait what is 1, when u remember make it a variable. 

        # this could probaby be handled prettier with like a dictionary of functions but I dont care
        # something is telling me I should have everything defined in init.
        if self.aggregator_type == 'flatten':
            self.aggregator = torch.flatten
            # this should compile but is unscientific
            # self.linear_layers = torch.nn.Sequential(torch.nn.Conv1d(1,1,33), torch.nn.Linear(self.out_height, self.num_classes)) # self-sabotage?
            self.linear_layers = torch.nn.Sequential(OrderedDict([('conv1',torch.nn.Conv1d(1,1,33)), ('dense1', torch.nn.Linear(self.out_height, self.num_classes))])) # self-sabotage?
            # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, self.out_height), torch.nn.Linear(self.out_height, self.num_classes)) # cheating?
        elif self.aggregator_type == 'lstm':
            self.aggregator = nn.LSTM(self.out_height, self.out_height, 1, batch_first=True)
        elif self.aggregator_type == 'mean':
            self.aggregator = torch.mean

         


        # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, self.num_classes))

        self.hook = self.linear_layers.register_forward_hook(get_features('feats'))
        if self.dropout:
            print('Using dropout')
            self.cnn_layers.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.linear_layers[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))
 

    def forward(self, x):
        #is this all on the comp. graph
        # print("1")
        frames = torch.empty((x.size(1), x.size(0), self.out_height)).type_as(x)

        # print("2")
        # print(x.dtype)
        x = x.unsqueeze(dim=1)
        x = self.transform(x)
        x = x.squeeze(dim=1)
        # print("3")
        # print(x.dtype)

        if self.show_image:
            image_to_print = x[0][0].clone().detach().unsqueeze(dim=0)
            self.logger.experiment.add_image("train image before transform", image_to_print )
            self.show_image = False
    
        for t in range(self.number_of_frames):
            frames[t] = self.fc(self.cnn_layers(x[:, t, :, :].unsqueeze(1)))
        frames = frames.permute((1,0,2)) # restore batch order
        # print("4")
        # print(frames.shape)
        
        # while the if statements were well intentioned, this is dumb. 
        if self.aggregator_type == 'flatten': 
            aggregated_frames = self.aggregator(frames, start_dim=1)
            aggregated_frames =  aggregated_frames.unsqueeze(dim=1)
            
        if self.aggregator_type == 'lstm':
            outputs, (hn, cn) = self.aggregator(frames)
            aggregated_frames = hn[-1]
            # I set the lstm up to output out-height

        if self.aggregator_type == 'mean':
           aggregated_frames  = self.aggregator(frames, dim=1)
           # now out height
        # print("Aggregating")
        # print(aggregated_frames.shape)
        # print("Final dense")
        logits = self.linear_layers(aggregated_frames)
        logits = logits.squeeze(dim=1)
        # print("logits shape:", logits.shape)
        # print(logits)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    
    def get_embedding_plot(self, points:torch.Tensor, labels:torch.Tensor, name:str) -> plt:
        """
        Get the an embedding from the final embedding (the decision space).
        """
        xs = []
        ys = []

        temp = np.transpose(points)
        xs = temp[0]
        ys = temp[1]
        
        colors = ['red','blue']

        np.random.seed(42)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(xs, ys, c=labels.argmax(dim=1).tolist(), cmap=matplotlib.colors.ListedColormap(colors))
    
        plt.title(name)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        return plt.gcf()

    def batch2points(self, l:list)-> torch.tensor:
        """ Helper function for dealing with intermediate data structures across pl.lightning. 
        Take a tensor that is a list-like object of tensor batches, and convert it to a set of points (N, sample.shape).
        Should depreciate

        returns points: (N, sample.shape)
         """
        return torch.cat(l,dim=0)

    def get_gmm_memberships(self, points: np.ndarray):
        print("entering the call to gmm")
        membership_probs = self.gmm.predict_proba(points)
        print(membership_probs.shape)
        return membership_probs

    def get_gmm_preds(self, points:torch.Tensor) -> torch.Tensor:
        """ 
        returns gmm preds in a one hot encoded tensor
        """
        print("entering get gmm preds")
        points = np.asanyarray(points, dtype=tuple)
        membership_probs = self.gmm.predict_proba(points)
        preds = membership_probs.argmax(axis=1)
        preds = torch.tensor(preds)
        preds = torch.nn.functional.one_hot(preds, num_classes=2)
        return preds

    def get_gmm_imprint(self, n=100):   
        """ Given a fitted gmm, get a sample of the distribution"""

        print("entering call to get gmm imprint")
        #TODO - dimension reduction. Right now its only hitting 2 of the frames * hidden_size tuple
        samples, labels = self.gmm.sample(n_samples=n)
        xxs = [x[0] for x in samples]
        yys = [x[1] for x in samples]
        plt.scatter(xxs, yys, c=labels)
        
        return plt.gcf()


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'pred': preds, 'target': labels, 'embeds': outputs, 'z': features['feats']}

    def training_step_end(self, outputs):
        self.train_accuracy(outputs['pred'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_loss', outputs['loss'], sync_dist=True)
        return outputs

    def train_epoch_end(self, outputs):
        #debugging too many open figs
        open_figs = plt.get_fignums()
        print("number of open figures:", len(open_figs))
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y) #???
        pred = logits.softmax(dim=-1)
        return {'loss':loss, 'pred': pred, 'target': y, 'embeds':logits}
        # return pred

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['pred'], outputs['target'])
        self.log('val_accuracy', self.val_accuracy)
        self.log('val_loss', outputs['loss'], sync_dist=True)
        return outputs

    def validation_epoch_end(self, outputs):
        embeds = [x['embeds'].detach().cpu() for x in outputs]
        points = self.batch2points(embeds)
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        targets = torch.cat(targets, dim=0)

        # TODO - reinstate
        plot = self.get_embedding_plot(points, targets, self.annotation)
        self.logger.experiment.add_figure("val embedding", plot)
        plt.close(fig=plot)
        return

    def test_step(self, batch, batch_idx):
        x, verbose_y = batch
        y = verbose_y['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.softmax(dim=-1)
        self.test_accuracy(pred, y)
        return {'loss':loss, 'pred': pred, 'target': y, 'embeds': y_hat, 'z': features}

    
    def test_epoch_end(self, outputs):

        embeds = [x['embeds'].detach().cpu() for x in outputs]
        points = self.batch2points(embeds)
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        targets = torch.cat(targets, dim=0)

        plot = self.get_embedding_plot(points, targets, self.annotation)
       
       # I truly don't know how to deal with this data type. this is the only way??
        Zs = [x['z']['feats'].detach().cpu() for x in outputs]
        points = self.batch2points(Zs)

        preds = [x['pred'].cpu().to('cpu') for x in outputs] 
        preds = torch.cat(preds, dim=0)
    

        print("shape of  one sample right before fit:", points[0].shape, len(points), " samples")
        # if len points.shape() == 3
        print(points.shape)
        points = points.squeeze(dim=1)
        print("last", points.shape)
        self.gmm = GMM(n_components=2, random_state=0).fit(points)
        print("fitted")
        gmm_preds = self.get_gmm_preds(points) # jesus 
        # here
     
        with open('features_out.npy', 'wb') as f:
            np.save(f, Zs)

        print("plotting...")
        plot = self.get_embedding_plot(points, targets, self.annotation)
        self.logger.experiment.add_figure("test embedding", plot)
        plt.close(fig=plot)

        print("gmm accuracy...")
        # by hand
        metric = torchmetrics.Accuracy(task="binary")
        gmm_acc = metric(gmm_preds, targets) 
        self.log('gmm_accuracy', gmm_acc)

        print("GMM fit metrics")
        print("bic" , self.gmm.bic(points))
        print("aic", self.gmm.aic(points))
        sample_plot = self.get_gmm_imprint()
        print("got imprint")
        self.logger.experiment.add_figure("gmm sample", sample_plot)
        plt.close(fig=sample_plot)
        print("added imprint")
        self.log('gmm bic', self.gmm.bic(points), sync_dist=True)
        print("added bic")

        acc = self.test_accuracy.compute()
        print('test_accuracy', acc)
        self.log('test accuracy', acc)

        return

