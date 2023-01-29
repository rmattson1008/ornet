
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

#please please please clean. 

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
            # ColorJiggle(0.1, 0.1, 0.1, 0.1, p=.5),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            # RandomChannelShuffle(p=0.75),
            # RandomThinPlateSpline(p=0.5),
            # RandomGaussianNoise
            # RandomPlasmaShadow
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
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model", dropout=False):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.number_of_frames = number_of_frames
        self.num_classes= num_classes
        # self.out_height = 32
        self.out_height = 8
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.gmm_accuracy = torchmetrics.Accuracy()
        self.label = label
        self.transform = DataAugmentation()
        self.dropout = dropout
        self.show_image = False
        self.gmm = None
        
    
        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay
        
        squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
        squeeze.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.cnn_layers = squeeze
        self.fc = torch.nn.Linear(1000, self.out_height)
        self.aggregator = torch.flatten
        self.flattened_frames_size = self.number_of_frames * 1 * self.out_height
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, self.num_classes))
  
        # self.fc = torch.nn.Sequential(torch.nn.Linear(1000, self.out_height), torch.nn.Linear(self.out_height, self.out_height))
        # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, 10), torch.nn.Linear(10, self.num_classes) )
        # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, 64),torch.nn.Linear(64, 32) , torch.nn.Linear(32, 16), torch.nn.Linear(16, self.num_classes))
        # self.linear_layers.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        # self.alt_linear_layer

        self.hook = self.linear_layers.register_forward_hook(get_features('feats'))
        if self.dropout:
            print('Using dropout')
            self.cnn_layers.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.linear_layers[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))
 

    def forward(self, x):
        #is this on the comp. graph
        frames = torch.empty((x.size(1), x.size(0), self.out_height)).type_as(x)

        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x = self.transform(x)
        # print(x.shape)
        x = x.squeeze(dim=1)
        # print(x.shape)

        if self.show_image:
            image_to_print = x[0][0].clone().detach().unsqueeze(dim=0)
            self.logger.experiment.add_image("train image before transform", image_to_print )
            self.show_image = False
    
        for t in range(self.number_of_frames):
            frames[t] = self.fc(self.cnn_layers(x[:, t, :, :].unsqueeze(1)))

        #TODO - why do I have to do this... 
        frames = frames.permute((1,0,2))
        aggregated_frames = self.aggregator(frames, start_dim=1)

        logits = self.linear_layers(aggregated_frames)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    
    def get_embedding_plot(self, embeddings, targets, preds, name):
        """
        Get the an embedding from the final embedding (the decision space).
        """
        print("getting plot...")
        xs = []
        ys = []

        labels = []
        preds = []

        # points = []

        points = self.batch2points(embeddings)
        temp = np.transpose(points)
        xs = temp[0]
        ys = temp[1]

        print("X shape", xs.shape)
        
        # TODO - use batch2points and detach correctly
        for batch in targets:
            labels.extend(batch.numpy())

        
        # preds = self.batch2points(preds)
        # or reshape
        # preds = [x.argmax(axis=0) for x in preds]

        for batch in preds:
            for encoding in batch: 
                label = encoding.argmax(axis=0)
                preds.append(label.item())
        
        colors = ['red','blue']

        np.random.seed(42)
        fig = plt.figure()
        ax = fig.add_subplot()
        # plt.plot(xs, ys, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(xs, ys, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    
        plt.title(name)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        return plt.gcf()

    def batch2points(self, l:torch.Tensor):
        """ Helper function for dealing with intermediate data structures across pl.lightning. 
        Take a tensor that is a list-like object of batches, and convert it to a set of points (N, sample.shape).

        returns points: (N, sample.shape)
         """
        
        # never do it like this
        # I think the pythonic way is to use reshape(-1)??
        # please redo before leaving this project

        points = []   
        for batch in l:
            points.extend(batch)
        points = np.stack(points)

        print("batch2points returned an array of shape", points.shape)
        return points

    def get_gmm_memberships(self, points: np.ndarray):
        print("entering the call to gmm")
        # print(points.shape)
        # print(points[0].shape)
        membership_probs = self.gmm.predict_proba(points)
        print(membership_probs.shape)
        return membership_probs

    def get_gmm_preds(self, points:np.ndarray):
        print("points", points.shape)
        membership_probs = self.gmm.predict_proba(points)
        print("Test", membership_probs.shape)
        preds =  membership_probs.argmax(axis=1)
        print(preds.shape)
        return preds

    def get_gmm_imprint(self, n=100):   
        """ Given a fitted gmm, get a sample of the distribution"""

        #TODO - dimension reduction. Right now its only hitting 2 of the frames * hidden_size tuple
        samples, labels = self.gmm.sample(n_samples=n)
        xxs = [x[0] for x in samples]
        yys = [x[1] for x in samples]
        plt.scatter(xxs, yys, c=labels)
        return plt.gcf()

    # def get_gmm_diff_plot(self, points):
    #     """
    #     So this is probably where I want to see where test videos or control videos fit in to the mm memberships. 
    #     It would be nice to project the memberships between -1 and 1, and plot how those change over time. 
    #     Not sure that the test dataset is ready for this tho...
    #     Got to review gmm outputs...
    #     I guess to start I could simply plot the membership on two axes
    #     Then switch to a view over time.
    # This code doesnt make any sennse
    #     """

    #     membership_probs = self.get_gmm_memberships(points)

    #     prob_diff = [p[0] - p[1] for p in membership_probs[:20]]

    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #     ax.scatter( range(0,len(prob_diff)), prob_diff)
    #     ax.plot(range(0,len(prob_diff)), prob_diff)

    #     plt.title("GMM membership")
    #     ax.set_ylabel('membership prob diff')
    #     ax.set_xlabel('time stamp')
    #     return plt.gcf()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print(inputs.shape)
        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'pred': preds, 'target': labels, 'embeds': outputs, 'z': features['feats']}

    def training_step_end(self, outputs):
        self.train_accuracy(outputs['pred'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_loss', outputs['loss'], sync_dist=True)
        return outputs

    def training_epoch_end(self, outputs):
    
        embeds = [x['embeds'].detach().cpu().numpy() for x in outputs]
        # get features
        # Zs = [x['z'].detach().cpu().numpy().reshape(-1) for x in outputs]
        Zs = [x['z'].detach().cpu().numpy() for x in outputs]

        points = self.batch2points(Zs) 

        # so ur getting a new model every time.
        print("right before fit:", points[0].shape, len(points), " samples")
        self.gmm = GMM(n_components=2, random_state=0).fit(points)
        print("GMM fit metrics")
        print("bic" , self.gmm.bic(points))
        print("aic", self.gmm.aic(points))
        sample_plot = self.get_gmm_imprint()
        self.logger.experiment.add_figure("gmm sample", sample_plot)
        self.log('gmm bic', self.gmm.bic(points),sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y) #???
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = logits.softmax(dim=-1)
        # print("val preds", preds)

        if self.gmm is not None:
            Zs = features['feats'].cpu().numpy()
            gmm_preds = self.get_gmm_preds(Zs)
            print(type(gmm_preds))

            self.gmm_accuracy(torch.Tensor(gmm_preds), y.detach().cpu()) # dont' use this 
        
        return {'loss':loss, 'pred': pred, 'target': y, 'embeds':logits}
        # return pred

    def validation_step_end(self, outputs):
        self.val_accuracy(outputs['pred'], outputs['target'])
        self.log('val_accuracy', self.val_accuracy)
        
        self.log('val_loss', outputs['loss'], sync_dist=True)
        return outputs

    def validation_epoch_end(self, outputs):
            #lets see what happens if we do this every time
        #I think this is a bad idea why not use detach? 
        embeds = [x['embeds'].detach().cpu().numpy() for x in outputs]
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        preds = [x['pred'].cpu().to('cpu') for x in outputs] 

        if self.gmm is not None:
            print("now computing accuracy")
           
            acc = self.gmm_accuracy.compute()
            self.log('gmm_accuracy', acc)

            print("Now plotting")
            plot = self.get_embedding_plot(embeds, targets, preds, self.label)
            self.logger.experiment.add_figure("val embedding", plot)
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
       
        embeds = [x['embeds'].detach().cpu().numpy() for x in outputs]
        Zs = [x['z']['feats'].detach().cpu().numpy() for x in outputs]

        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        preds = [x['pred'].cpu().to('cpu') for x in outputs] 
    
        print("Features", len(Zs))
        print("Features", Zs[0].shape)
        Zs = self.batch2points(Zs)

        with open('features_out.npy', 'wb') as f:
            np.save(f, Zs)

        plot = self.get_embedding_plot(embeds, targets, preds, self.label)
        self.logger.experiment.add_figure("test embedding", plot)

        acc = self.test_accuracy.compute()
        print('test_accuracy', acc)
        self.log('test accuracy', acc)
        return






# class LSTM(nn.Module):
#     def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="lstm_model"):
#         super().__init__()
#         self.register_buffer("sigma", torch.eye(3))
#         self.number_of_frames = number_of_frames
#         self.num_classes= num_classes
#         self.out_height = 128
#         self.label = label
#         self.hidden_layer_size = 128

#         self.lstm = nn.LSTM(self.out_height,self.hidden_layer_size, 1)

    
#     def forward(self, frames):
#         print(frames.shape)

#         frames = frames.flatten(start_dim=2) #flattening the 2d image into one vector

#         #todo, switch if bi
#         h0 = torch.randn(1, frames.size(1), self.hidden_layer_size).type_as(x)
#         c0 = torch.randn(1,frames.size(1), self.hidden_layer_size).type_as(x)
#         out, (hn, cn) = self.lstm(frames)
#         out = out.permute((1,0,2))#why??
#         print(out.shape)

#         out = out.flatten(start_dim=1) #why? do I really need this?
#         print(out.shape)
#         # print('applied classifier')
#         # should be one point in space from 3
#         return out


"""keep in mind a LightningModule is a nn.Module, so whenever you define a nn.Module as 
attribute to a LightningModule in the __init__ function, this module will end being registered
 as a sub-module to the parent pytorch lightning module."""



#TODO - now the imprints wont work


