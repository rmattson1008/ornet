import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


from data_utils import DynamicVids, RoiTransform
from models import CNN_LSTM
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser 

from matplotlib import pyplot as plt

def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    lr = args.lr
    epochs = args.epochs
    # optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()


    model.to(device)
    for epoch in range(epochs):  
        print("Epoch", epoch)
        model.train()
        training_loss = 0.0

        #init hidden state tracker? Does it matter if you batch_first or not?? 
        
        #not sure how this works in for a cnn + lstm, just lstm
        #maybe init a 
        # state_h, state_c = model.init_state() #why do I need this. OK ignore state for now, use it to deal with batch initializaion later
        #TODO still confused on sequnce length(time steps) vs number of lstm layers. ig its layers per timestep, like you could have multiple? (think bidirectional stack lstm)
        #Maybe jus say num_layers = 1 for now 
        for data in train_dataloader: 
            # get the inputs; data is a list of [inputs, labels]
            # print(data.shape)
            inputs, labels = data[0].to(device), data[1].to(device)
            print(labels)
            inputs = inputs.float() # shouldn't stay on this step.
            print("input shape", inputs.shape)
            #TODO resize... 
            #should be [batch_size, L(time_steps), input_size]. I dont know if forward handles.... lets just try.... 

            # zero the parameter gradients
            optimizer.zero_grad()

            # todo pass hidden state info between batches. 
            outputs, _ = model(inputs)
            #state is returned from nn.LSTM, its (h_n, c_n). ex h_n should be [(1|2) * num_layers, H_out]. 
            #Ok so the other person wanted to save [num layers, t, hiddem_size ]. basically the same but extended for every hidden state t

            loss = criterion(outputs, labels) # uhhhh make sure outputs is correct y_pred
            loss.backward()
            optimizer.step()
            training_loss += loss.item()


        # model.eval()
        # valid_loss = 0.0
        # for inputs, labels in val_dataloader:
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     inputs = inputs.float() # shouldn't stay on this step. 

        #     pred = model(inputs)
        #     loss = criterion(pred, labels)
        #     valid_loss += loss.item()


        # print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
        #      \t\t Validation Loss: {valid_loss / len(val_dataloader )}')
        #TODO - stop training when Val drops?

    return



def get_dataloaders(args, display_images=False):
    """
    Access ornet dataset, apply any necessary transformations to images, 
    split into train/test/validate, and return dataloaders
    """

    #
    if args.roi:
        print("Using ROI inputs")
        transform = transforms.Compose([RoiTransform(window_size=(28,28))])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=28)])
    # TODO - normalize??
    # dataset = FramePairDataset(args.input_dir, class_types=args.classes, transform=transform)
    dataset = DynamicVids(args.input_dir, class_types=args.classes, transform=transform) # nt working :/ 

    print("dataset", len(dataset))
    
    if display_images:
        # This code is here if you are like me and need to see to believe your data exists
        for i, ((img, _), __) in enumerate(dataset):
            if i % 20 == 0:
                plt.imshow(img)
                plt.show()
    
    ## don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = int(train_split * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = int((len(dataset) - train_size) / 2)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(69))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # if args.weighted_samples:
    #     sampler = get_weighted_sampler(train_dataset, dataset)
    #     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    # else:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader, val_dataloader




if __name__ == "__main__":
    print("Main")
    # going to use the same parser manager for cnn and rnn
    args, _ = make_parser()

    num_lstm_layers = 200 #too many?
    # lstm_hidden_size = -1 # if we decide to have a different hidden space size then must add to model

    model = CNN_LSTM(num_lstm_layers)

    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'
    # device = 'cpu'
    print("device", device)
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)
    print("Train_size", len(train_dataloader))
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)
    
    # if args.test:
    #     #TODO - if no model, load saved model
    #     print("Testing")
    #     test(args, model, test_dataloader)




# TODO - main idea is getting the cnn to deal with reps, turn it into a sequence of frame representations. right not its trying to swallow too much. 
