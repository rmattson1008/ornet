
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
import kornia.augmentation as K
import matplotlib.pyplot as plt
import matplotlib
import logging
from sklearn.mixture import GaussianMixture as GMM

#please please please clean. 
    
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
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
# tensor([0.0024, 0.0024, 0.0024, 0.0024, 0.0024]) tensor([0.0149, 0.0149, 0.0150, 0.0150, 0.0150])
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
        self.out_height = 8
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.label = label
        self.transform = DataAugmentation()
        self.dropout = dropout
        self.show_image = True
        self.gmm = None
        
      
        # self.val_bin_accuracy = torchmetrics.BinaryAccuracy()

        # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # self.logger = init_logger("dnn", notebook=True)
        # self.logger = logging.getLogger("pytorch_lightning.core")

    # configure logging on module level, redirect to file
        # self.tensorboard = self.logger.experiment
       

        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay

        # self.save_hyperparameters()
        # self.device = device

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # inception_v3
        

        # resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # # resnet.fc = torch.nn.Linear(2048, self.out_height,bias=.6)
        # resnet.fc = torch.nn.Linear(512, self.out_height)
        # # resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        # if self.dropout:
        #     resnet.avgpool.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        #     resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        

        squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
        squeeze.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = torch.nn.Linear(1000, self.out_height)
        self.aggregator = torch.flatten
  


        # self.fc = torch.nn.Sequential(torch.nn.Linear(1000, self.out_height), torch.nn.Linear(self.out_height, self.out_height))
    
        # 

        self.cnn_layers = squeeze
        # self.cnn_layers = vgg
        self.flattened_frames_size = self.number_of_frames * 1 * self.out_height
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, self.num_classes))
        # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, 10), torch.nn.Linear(10, self.num_classes) )
        # self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.flattened_frames_size, 64),torch.nn.Linear(64, 32) , torch.nn.Linear(32, 16), torch.nn.Linear(16, self.num_classes))
        # self.linear_layers.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        # self.alt_linear_layer
        if self.dropout:
            print('Using dropout')
            self.cnn_layers.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            # self.linear_layers[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))
 


    def forward(self, x):
        # I want to classify like 3-5 frames
        frames = torch.empty((x.size(1), x.size(0), self.out_height)).type_as(x)

        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x = self.transform(x)
        # print(x.shape)
        x = x.squeeze(dim=1)
        # print(x.shape)


        if self.show_image:
            print("printing")
            image_to_print = x[0][0].clone().detach().unsqueeze(dim=0)
            # self.logger.experiment.add_image("train image before transform", image_to_print )
            self.show_image = False
    
        #??? why would u write it like this??
        for t in range(self.number_of_frames):
            frames[t] = self.fc(self.cnn_layers(x[:, t, :, :].unsqueeze(1)))
        # frames = self.cnn_layers(x)

        frames = frames.permute((1,0,2))
        # aggregate frames
        fcs_vector = self.aggregator(frames, start_dim=1)

    
        # print("flat shape" ,fcs_vector.shape)

        # print("")
        logits = self.linear_layers(fcs_vector)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    
    def get_embedding_plot(self, embeddings, targets, preds, name):
        """
        truly some shitty code
        """
        print("getting plot...")
        xs = []
        ys = []
        labels = []
        preds = []

        # points = []

        points = self.batch2array(embeddings)
        temp = np.transpose(points)
        xs = temp[0]
        ys = temp[1]

        print("X shape", xs.shape)
        
        # TODO - use batch2array and detach correctly
        for batch in targets:
            labels.extend(batch.numpy())

        
        # preds = self.batch2array(preds)
        # preds = [x.argmax(axis=0) for x in preds]
        for batch in preds:
            for fuc in batch: 
                fuc = fuc.argmax(axis=0)
                preds.append(fuc.item())
        
        colors = ['red','blue']

        np.random.seed(42)
        fig = plt.figure()
        ax = fig.add_subplot()
        # plt.plot(xs, ys, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(xs, ys, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    
        plt.title(name)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        membership_probs = self.get_gmm_memberships(points)
    
        gmm_pred = [np.argmax(mm) for mm in membership_probs]

        assert len(membership_probs) == len(labels)

        # print("val accuracy", len([labels[i] for i in range(0, len(labels)) if labels[i] == preds[i]]) / len(labels))
        # print("gmm accuracy", len([labels[i] for i in range(0, len(labels)) if labels[i] == gmm_pred[i]]) / len(labels))
        
        return plt.gcf()

    def batch2array(self, l):
        #never do it like this
        points = []   
        print("in batch")
        print(type(l)) 
        for batch in l:
            #Todo do I need to permute somehow?
            points.extend(batch)
        # print(points[0].shape)
        points = np.stack(points)
        print("batch2array returned an array of shape", points.shape)
        # print(points)
        # exit()
        return points

    def get_gmm_memberships(self, points):
        print("entering the call to gmm")
        membership_probs = self.gmm.predict_proba(points)
        print(membership_probs.shape)
        return membership_probs


    def get_gmm_imprint(self, n=50):
        
        samples, labels = self.gmm.sample(n_samples=n)
        xxs = [x[0] for x in samples]
        yys = [x[1] for x in samples]
        plt.scatter(xxs, yys, c=labels)
        return plt.gcf()

    def get_gmm_diff_plot(self, points):
        """
        So this is probably where I want to see where test videos or control videos fit in to the mm memberships. 
        It would be nice to project the memberships between -1 and 1, and plot how those change over time. 
        Not sure that the test dataset is ready for this tho...
        Got to review gmm outputs...
        I guess to start I could simply plot the membership on two axes
        Then switch to a view over time.
        """

        membership_probs = self.get_gmm_memberships(points)

        prob_diff = [p[0] - p[1] for p in membership_probs[:20]]

        # probax1 = [p[0] for p in membership_probs]
        # probax2 = [p[1] for p in membership_probs]

        # colors = ['red','blue']
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter( range(0,len(prob_diff)), prob_diff)
        ax.plot(range(0,len(prob_diff)), prob_diff)

        plt.title("GMM membership")
        ax.set_ylabel('membership prob diff')
        ax.set_xlabel('time stamp')
        return plt.gcf()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print(inputs.shape)
        outputs = self(inputs)
        preds = outputs.softmax(dim=-1)

        loss = F.cross_entropy(outputs, labels)
        # return {"loss": loss}
        
        return {'loss': loss, 'pred': preds, 'target': labels, 'embeds': outputs}
        # Yeah so it knows the loss?? ? idk. anyway 

    def training_step_end(self, outputs):
        self.train_accuracy(outputs['pred'], outputs['target'])
        self.log('train_accuracy', self.train_accuracy)
        self.log('train_loss', outputs['loss'], sync_dist=True)
        return outputs

    def training_epoch_end(self, outputs):
    
        embeds = [x['embeds'].detach().cpu().numpy() for x in outputs]
        # print(embeds.shape)

        points = self.batch2array(embeds)
        print(type(points))
        # so ur getting a new model every time.
        print("right before fit:", points[0].shape, len(points))
        self.gmm = GMM(n_components=2, random_state=0).fit(points)
        print("GMM fit metrics")
        print("bic" , self.gmm.bic(points))
        print("aic", self.gmm.aic(points))
        sample_plot = self.get_gmm_imprint()
        self.logger.experiment.add_figure("gmm sample", sample_plot)
        self.log('gmm bic', self.gmm.bic(points),sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        
        return {'loss':loss, 'pred': pred, 'target': y, 'embeds':y_hat}
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
            plot = self.get_embedding_plot(embeds, targets, preds, self.label)
            self.logger.experiment.add_figure("val embedding", plot)
        return

    def test_step(self, batch, batch_idx):
        x, verbose_y = batch
        y = verbose_y['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        pred = y_hat.softmax(dim=-1)
        # print("val preds", preds)
        self.test_accuracy(pred, y)
        # print(verbose_y)




        return {'loss':loss, 'pred': pred, 'target': y, 'embeds': y_hat}
        # return pred
    
    def test_epoch_end(self, outputs):
       
        embeds = [x['embeds'].detach().cpu().numpy() for x in outputs]
        targets = [x['target'].cpu().to('cpu') for x in outputs] 
        preds = [x['pred'].cpu().to('cpu') for x in outputs] 
    

        plot = self.get_embedding_plot(embeds, targets, preds, self.label)
        self.logger.experiment.add_figure("test embedding", plot)

        #TODO - HERE
        points = self.batch2array(embeds)
        mm_plot = self.get_gmm_diff_plot(points)
        # print(mm_out)
        self.logger.experiment.add_figure("mm", mm_plot)


        acc = self.test_accuracy.compute()
        print('test_accuracy', acc)
        self.log('test accuracy', acc)
        return









































class LSTM(nn.Module):
    def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model", dropout=False):
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
        self.dropout = dropout
        self.show_image = True
        # self.val_bin_accuracy = torchmetrics.BinaryAccuracy()

        self.lr = learning_rate
        print("using learning rate ", self.lr)
        self.wd = weight_decay

        # self.save_hyperparameters()
        # self.device = device

        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
        print(squeeze)
        # resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        squeeze.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # resnet.fc = torch.nn.Linear(2048, self.out_height,bias=.6)
        # resnet.fc = torch.nn.Linear(512, self.out_height)
        # squeeze.classifier = torch.nn.Linear(512, self.out_height)
        self.fc = torch.nn.Linear(1000, self.out_height)
        if self.dropout:
            # resnet.avgpool.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            # resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))

            squeeze.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        print(squeeze)

        # self.cnn_layers = resnet
        self.cnn_layers = squeeze
        self.lstm = nn.LSTM(self.out_height,self.hidden_layer_size, 1)

        # self.flattened_frames_size = self.number_of_frames * 1 * self.out_height.... what is out size?
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_layer_size * self.number_of_frames, 10), torch.nn.Linear(10, self.num_classes) )
        if self.dropout:
            # pass
            print('Using dropout')
            self.linear_layers[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))
       
    def forward(self, x):
        # I want to classify like 3-5 frames
        frames = torch.empty((x.size(1), x.size(0) , self.out_height)).type_as(x)

        # This chunk is pretty gross, should just keep dummy channel dim
        # print(x.shape)
        # self.logger.experiment.add_image("train image before transform", x[0][0].unsqueeze(dim=0))
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x = self.transform(x)
        # print(x.shape)
        x = x.squeeze(dim=1)

        # self.logger.experiment.add_image("train image", x[0][0].unsqueeze(dim=0))
        if self.show_image:
            # print("printing")
            image_to_print = x[0][0].clone().detach().unsqueeze(dim=0)
            self.logger.experiment.add_image("train image before transform",image_to_print )
            self.show_image = False
        for t in range(self.number_of_frames):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
            # print(x[:, t, :, :].unsqueeze(1).shape)
            frames[t] =  self.fc(self.cnn_layers(x[:, t, :, :].unsqueeze(1)))

        frames = frames.flatten(start_dim=2)

        #todo, switch if bi
        h0 = torch.randn(1, frames.size(1), self.hidden_layer_size).type_as(x)
        c0 = torch.randn(1,frames.size(1), self.hidden_layer_size).type_as(x)
        out, (hn, cn) = self.lstm(frames)
        out = out.permute((1,0,2))

        out = out.flatten(start_dim=1)
        # print(out.shape)

        logits = self.linear_layers(out)
        # print('applied classifier')
        return logits







# class CnnLSTM_Module(pl.LightningModule):
#     def __init__(self, number_of_frames=5, num_classes=2, learning_rate=0.00001, weight_decay=0, label="model", dropout=False):
#         super().__init__()
#         self.register_buffer("sigma", torch.eye(3))
#         self.number_of_frames = number_of_frames
#         self.num_classes= num_classes
#         self.out_height = 128
#         self.train_accuracy = torchmetrics.Accuracy()
#         self.val_accuracy = torchmetrics.Accuracy()
#         self.test_accuracy = torchmetrics.Accuracy()
#         self.label = label
#         self.transform = DataAugmentation()
#         self.hidden_layer_size = 128
#         self.dropout = dropout
#         self.show_image = True
#         # self.val_bin_accuracy = torchmetrics.BinaryAccuracy()

#         self.lr = learning_rate
#         print("using learning rate ", self.lr)
#         self.wd = weight_decay

#         # self.save_hyperparameters()
#         # self.device = device

#         # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
#         # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
#         squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
#         print(squeeze)
#         # resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         squeeze.features[0] = Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # resnet.fc = torch.nn.Linear(2048, self.out_height,bias=.6)
#         # resnet.fc = torch.nn.Linear(512, self.out_height)
#         # squeeze.classifier = torch.nn.Linear(512, self.out_height)
#         self.fc = torch.nn.Linear(1000, self.out_height)
#         if self.dropout:
#             # resnet.avgpool.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
#             # resnet.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
#             squeeze.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
#         print(squeeze)

#         # self.cnn_layers = resnet
#         self.cnn_layers = squeeze
#         self.lstm = nn.LSTM(self.out_height,self.hidden_layer_size, 1)

#         # self.flattened_frames_size = self.number_of_frames * 1 * self.out_height.... what is out size?
#         self.linear_layers = torch.nn.Sequential(torch.nn.Linear(self.hidden_layer_size * self.number_of_frames, 10), torch.nn.Linear(10, self.num_classes) )
#         if self.dropout:
#             # pass
#             print('Using dropout')
#             self.linear_layers[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))
       

#     def forward(self, x):
#         # I want to classify like 3-5 frames
#         frames = torch.empty((x.size(1), x.size(0) , self.out_height)).type_as(x)

#         # This chunk is pretty gross, should just keep dummy channel dim
#         # print(x.shape)
#         # self.logger.experiment.add_image("train image before transform", x[0][0].unsqueeze(dim=0))
#         x = x.unsqueeze(dim=1)
#         # print(x.shape)
#         x = self.transform(x)
#         # print(x.shape)
#         x = x.squeeze(dim=1)

#         # self.logger.experiment.add_image("train image", x[0][0].unsqueeze(dim=0))
#         if self.show_image:
#             # print("printing")
#             image_to_print = x[0][0].clone().detach().unsqueeze(dim=0)
#             self.logger.experiment.add_image("train image before transform",image_to_print )
#             self.show_image = False
#         for t in range(self.number_of_frames):
#             # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
#             # print(x[:, t, :, :].unsqueeze(1).shape)
#             frames[t] =  self.fc(self.cnn_layers(x[:, t, :, :].unsqueeze(1)))

#         frames = frames.flatten(start_dim=2)

#         #todo, switch if bi
#         h0 = torch.randn(1, frames.size(1), self.hidden_layer_size).type_as(x)
#         c0 = torch.randn(1,frames.size(1), self.hidden_layer_size).type_as(x)
#         out, (hn, cn) = self.lstm(frames)
#         out = out.permute((1,0,2))

#         out = out.flatten(start_dim=1)
#         # print(out.shape)

#         logits = self.linear_layers(out)
#         # print('applied classifier')
#         return logits

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
#         #TODO- use my args
#         return optimizer

#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#         # print(inputs.shape)
#         outputs = self(inputs)
#         preds = outputs.softmax(dim=-1)

#         loss = F.cross_entropy(outputs, labels)
        
#         return {'loss': loss, 'pred': preds, 'target': labels}


#     def training_step_end(self, outputs):
#         self.train_accuracy(outputs['pred'], outputs['target'])
#         self.log('train_accuracy', self.train_accuracy)
#         self.log('train_loss', outputs['loss'], sync_dist=True)
#         return outputs


#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
#         pred = y_hat.softmax(dim=-1)
#         # print("val preds", preds)
        
#         return {'loss':loss, 'pred': pred, 'target': y}
#         # return pred

#     def validation_step_end(self, outputs):
#         self.val_accuracy(outputs['pred'], outputs['target'])
#         self.log('val_accuracy', self.val_accuracy)
#         self.log('val_loss', outputs['loss'], sync_dist=True)

#     def test_step(self, batch, batch_idx):
#         x, verbose_y = batch
#         y = verbose_y['label']
#         print("my y", y)

#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         # self.log('validation_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
#         pred = y_hat.softmax(dim=-1)
#         # print("val preds", preds)

#         #if wrong, print video and timestamp to buffer. 


#         self.test_accuracy(pred, y)
#         return {'loss':loss, 'pred': pred, 'target': y, 'embed': y_hat}
#         # return pred
    
#     def test_epoch_end(self, outputs):
       
#         embeds = [x['embed'].cpu().to('cpu') for x in outputs] 
#         targets = [x['target'].cpu().to('cpu') for x in outputs] 
#         preds = [x['pred'].cpu().to('cpu') for x in outputs] 
      
#         plot = get_embedding_plot(embeds, targets, preds, self.label)


#         # print("saving embeddings")
#         # with open("embeddings/" + self.label +"_embeddings.pkl", "wb") as fp:   #Pickling
#         #     pickle.dump(embeds, fp)
#         # with open("embeddings/" + self.label +"_targets.pkl", "wb") as fp:   #Pickling
#         #     pickle.dump(targets, fp)
#         # with open("embeddings/" + self.label +"_preds.pkl", "wb") as fp:   #Pickling
#             # pickle.dump(preds, fp)
#         # self.test_accuracy(outputs['pred'], outputs['target'])


#         self.logger.experiment.add_figure("test embedding", plot)
#         print('test_accuracy', self.test_accuracy.compute())
#         return



