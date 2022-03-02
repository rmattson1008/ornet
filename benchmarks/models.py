import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d


class Model_0(Module):
    

    def __init__(self):
        super(Model_0, self).__init__()


        self.cnn_layers = Sequential(
            # Convolution 1
            Conv2d(2,2,3,padding=1), # my god I do not know how to select params
            BatchNorm2d(2),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2),

            # Convolution 2
            #expand feature maps
            Conv2d(2,4,3, padding=1), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2)
        )

        # should be length of unwound channels * feature map dims
        db_size = 4 * 7 * 7
        # one layer to classify
        self.linear_layers = Sequential(Linear(db_size, 3))


    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x