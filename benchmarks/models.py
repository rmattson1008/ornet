from typing import OrderedDict
import torch
from collections import OrderedDict 
import torch.nn as nn
from torch.nn import Linear, ReLU, ELU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d, LeakyReLU


# 2 blocks, expands feature maps immediately and has 2 linear layers
class BaseCNN(Module):

    def __init__(self):
        super().__init__()
        self.rep_out = OrderedDict()
        # self.hooks = [] # only taking 1 hook at final layer

        self.cnn_layers = Sequential(
            # Convolution 1
            #expand feature maps
            Conv2d(2,4,3), 
            BatchNorm2d(4),
            LeakyReLU(inplace=True),
            MaxPool2d(3, stride=1),

            # Convolution 2
            Conv2d(4,4,3), 
            BatchNorm2d(4),
            LeakyReLU(inplace=True),
            MaxPool2d(3, stride=1)
        )
        #TODO check expansion

        # should be length of unwound channels * feature map dims
                cnn_out_size = 4 * 20 * 20
        
        self.final_rep_layer = nn.Sequential(Linear(cnn_out_size, 400), nn.ELU(), Linear(400, 10) , nn.ELU())
        # self.final_rep_layer = nn.Sequential(Linear(cnn_out_size, 1024), nn.ELU(), Linear(1024, 512) , nn.ELU(), Linear(512, 256) , nn.ELU(), Linear(256, 128) , nn.ELU(),  Linear( 128, 10) , nn.ELU())
        self.fc = Linear(10,3)


       
        self.hook = self.final_rep_layer.register_forward_hook(self.forward_hook("embeddings10"))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.rep_out[layer_name] = output
        return hook

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.final_rep_layer(x)
        x = self.fc(x)
        return x, self.rep_out
    
      
      
  
class CNN_LSTM(Module):

    def __init__(self, num_lstm_layers, bidirectional=False):
        super(CNN_LSTM, self).__init__()
        # self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional

        self.cnn_layers = Sequential(
            # Convolution 1
            #expand feature maps
            Conv2d(1,4,3), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(3, stride=1),

            # Convolution 2
            Conv2d(4,4,3), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(3, stride=1)
        )
        cnn_representation_size = 4 * 20 * 20

        self.lstm_input_size = cnn_representation_size
        self.lstm_hidden_size = cnn_representation_size

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_lstm_layers, bidirectional=self.bidirectional)
        self.linear_layers = Sequential(Linear(self.lstm_hidden_size, 3))

    # def forward(self, x, prev_state):

    def forward(self, x):
        hidden_state = None  # maybe actually init it, or use previous
        for t in range(x.size(1)):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well.
            frames = self.cnn_layers(x[:, t, :, :, :])
            frames = torch.flatten(frames, start_dim=1)
            out, hidden_state = self.lstm(frames, hidden_state) # ???? this make no sense tbh
            # we are not saving the sequence of hidden states for one sample
            # nor are we sharing hidden states over batches

        logits = self.linear_layers(out)
        return logits, hidden_state
      

# class VGG_Model(Module):
    
#     def __init__(self):
#         super().__init__()


#         self.cnn_layers = Sequential(
#             # BLock 1
#             Conv2d(2,64,3,padding=1), 
#             LeakyReLU(inplace=True),
#             Conv2d(64,128,3,padding=1), 
#             LeakyReLU(inplace=True),
#             MaxPool2d(2, stride=2),
#             # Block 2
#             Conv2d(128,256,3,padding=1), 
#             LeakyReLU(inplace=True),
#             Conv2d(256,256,3,padding=1), 
#             LeakyReLU(inplace=True),
#             MaxPool2d(2, stride=2),
#         )
#         cnn_representation_size = 4 * 20 * 20

#         # should be length of unwound channels * feature map dims
#         db_size = 256 * 7 * 7
#         # one layer to classify
#         self.linear_layers = Sequential(Linear(db_size, 1024), nn.ELU(), Linear(1024, 512), nn.ELU(), Linear(512, 10), nn.ELU())
#         self.fc = Linear(10,3)

# # def forward():

      
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.LeakyReLU()(self.bn1(self.conv1(input)))
        input = nn.LeakyReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.LeakyReLU()(input)


class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.rep_out = OrderedDict()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc = torch.nn.Linear(512, outputs)

        self.final_rep_layer = nn.Sequential(Linear(512, 256), nn.LeakyReLU(), Linear(256, 128), nn.LeakyReLU(), Linear(128, 10), nn.LeakyReLU()) 
        self.fc = Linear(10,outputs)

        self.hook = self.final_rep_layer.register_forward_hook(self.forward_hook("embeddings10"))
        # probably dont need self.hook... 


    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.rep_out[layer_name] = output
        return hook

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        # input = torch.flatten(input)
        input = input.view(input.size(0), -1)
        input = self.final_rep_layer(input)
        input = self.fc(input)

        return input, self.rep_out
