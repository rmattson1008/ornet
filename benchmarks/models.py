from typing import OrderedDict
import torch
from collections import OrderedDict 
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d, LeakyReLU


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
        
        self.final_rep_layer = Linear(cnn_out_size, 10)
        self.fc = Linear(10,3)

       
        self.hook = self.final_rep_layer.register_forward_hook(self.forward_hook("embedding10"))

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




# Losely styled after vgg model
class VGG_Model(Module):
    
    def __init__(self):
        super().__init__()


        self.cnn_layers = Sequential(
            # BLock 3
            Conv2d(2,64,3,padding=1), 
            LeakyReLU(inplace=True),
            Conv2d(64,128,3,padding=1), 
            LeakyReLU(inplace=True),
            MaxPool2d(2, stride=2),
            # Block 2
            Conv2d(128,256,3,padding=1), 
            LeakyReLU(inplace=True),
            Conv2d(256,256,3,padding=1), 
            LeakyReLU(inplace=True),
            MaxPool2d(2, stride=2),
        )

        # should be length of unwound channels * feature map dims
        db_size = 256 * 7 * 7
        # one layer to classify
        self.linear_layers = Sequential(Linear(db_size, 3))


    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



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

        self.final_rep_layer = Linear(512, 10)
        self.fc = Linear(10,outputs)

        self.hook = self.final_rep_layer.register_forward_hook(self.forward_hook("embedding10"))
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
