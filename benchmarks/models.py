import torch
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d


class Model_0(Module):
    

    def __init__(self):
        super(Model_0, self).__init__()


        self.cnn_layers = Sequential(
            # Convolution 1
            Conv2d(2,2,3,padding=1), 
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


import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Softmax, Module, BatchNorm2d

# very simlar to the first, just expands feature maps immediately and has 2 linear layers
class Model_1(Module):
    

    def __init__(self):
        super(Model_1, self).__init__()


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

# styled after vgg model
class VGG_Model(Module):
    
    def __init__(self):
        super(VGG_Model, self).__init__()


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


class CNN_LSTM(Module):
    
    def __init__(self, num_lstm_layers, bidirectional=False):
        super(CNN_LSTM, self).__init__()
        # self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional

        self.cnn_layers = Sequential(
            Conv2d(1,4,3,padding=1), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2),
            # Convolution 2
            Conv2d(4,4,3, padding=1), 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2)
            )
        cnn_representation_size = 4 * 7 * 7
        

        self.lstm_input_size = cnn_representation_size
        self.lstm_hidden_size = cnn_representation_size

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, bidirectional=self.bidirectional)
        self.linear_layers = Sequential(Linear(self.lstm_hidden_size, 3)) # Really??? 


        

    # def forward(self, x, prev_state):
    def forward(self, x):
        
        hidden_state = None # maybe actually init it, or use previous
        for t in range(x.size(1)):
            # with torch.no_grad(): # i think we want to unfreeze the cnn. everyone else doing this uses pretrained cnn oh well. 
            frames = self.cnn_layers(x[:,t, :, :, :])  
            frames = torch.flatten(frames, start_dim=1)
            out, hidden_state = self.lstm(frames, hidden_state)  
            # we are not saving the sequence of hidden states for one sample
            # nor are we sharing hidden states over batches

        logits = self.linear_layers(out)
        return logits, hidden_state



