import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d


# 2 blocks, expands feature maps immediately and has 2 linear layers
class BaseCNN(Module):
    

    def __init__(self):
        super().__init__()


        self.cnn_layers = Sequential(
            # Convolution 1
            #expand feature maps
            Conv2d(2,4,3,padding=1), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2),

            # Convolution 2
            Conv2d(4,4,3, padding=1), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2)
        )

        # should be length of unwound channels * feature map dims
        db_size = 4 * 7 * 7
        # one layer to classify
        self.linear_layers = Sequential(
            Linear(db_size, 10),
            Linear(10,3))


    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x




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