from typing import OrderedDict
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d
from collections import OrderedDict 

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
            ReLU(inplace=True),
            MaxPool2d(3, stride=1),

            # Convolution 2
            Conv2d(4,4,3), 
            BatchNorm2d(4),
            ReLU(inplace=True),
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
            ReLU(inplace=True),
            Conv2d(64,128,3,padding=1), 
            ReLU(inplace=True),
            MaxPool2d(2, stride=2),
            # Block 2
            Conv2d(128,256,3,padding=1), 
            ReLU(inplace=True),
            Conv2d(256,256,3,padding=1), 
            ReLU(inplace=True),
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

# class ResNet18(module):
#     def __init__(self):
#         super().__init__()

#     # Im trying to think if theres any real adjustments that need to be made... seems to accesp whatever input
#     #rly just copy pasta
#     def forward():

