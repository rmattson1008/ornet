import torch
from torch.utils.data import Dataset, DataLoader, get_accept_list
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


from data_utils import DynamicVids, RoiTransform
from models import CNN_LSTM
from sklearn.metrics import confusion_matrix
import numpy as np
from parsing_utils import make_parser

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import os
import math
from tqdm import tqdm
from time import sleep

from torch.utils.tensorboard import SummaryWriter
from itertools import product


def train(args, model, train_dataloader, val_dataloader, device='cpu'):
    # lr = args.lr
    epochs = args.epochs
    optimizer = Adam(model.parameters(), lr=args.lr)
    # optimizer = SGD(model.parameters(), lr=args.lr)
    print(optimizer)
    criterion = CrossEntropyLoss()
    comment = f' batch_size = {args.batch_size} lr = {args.lr} shuffle = {args.shuffle} epochs = {args.epochs} glob_lstm_adam'
    tb = SummaryWriter(comment=comment)

    # train_losses = []
    val_losses = []

    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()

        train_loss = 0.0
        val_loss = 0.0
        total_correct = 0.0
        val_total_correct = 0.0
        train_acc = 0.0
        val_acc = 0.0
        num_batches_used = 0.0 

        for batch_idx, data in enumerate(train_dataloader):
            # inputs, labels = data
            inputs, labels = get_augmented_batch(data)
            inputs, labels = inputs.to(device), labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            real_total_correct = outputs.argmax(dim=1).eq(labels).sum().item()
            train_acc += real_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        train_loss = train_loss / num_batches_used
        train_acc = train_acc / num_batches_used * 100

        model.eval()
        num_batches_used = 0.0 
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds, _ = model(inputs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            val_total_correct += preds.argmax(dim=1).eq(labels).sum().item()
            real_val_total_correct = preds.argmax(dim=1).eq(labels).sum().item()
            val_acc += real_val_total_correct / len(labels)
            num_batches_used = batch_idx + 1

        val_loss = val_loss / num_batches_used
        val_acc = val_acc / num_batches_used * 100
  

        tb.add_scalar("AvgTrainLoss", train_loss, epoch)
        tb.add_scalar("AvgValLoss", val_loss, epoch)
        tb.add_scalar("TrainAccuracy", train_acc, epoch)
        tb.add_scalar("ValAccuracy", val_acc, epoch)
        tb.add_scalar("TotalCorrect", total_correct, epoch)
        tb.add_scalar("ValTotalCorrect", val_total_correct, epoch)


        # save the model at lowest val loss score and continue training
        # the val loss curve is not incredibly smooth so I dont want to risk a local optimum
        val_losses.append(val_loss)
        if val_loss == np.min(val_losses):
            best_accuracy = val_acc
            if args.save_model:
                save_path = args.save_model + "_" + str(args.lr) + "_" +str(args.batch_size) + "_" +str(args.shuffle) + "_" +str(args.epochs)  + ".pth"
                # print("Saving model")
                torch.save(
                    {'epoch': epoch + 1,
                     'state_dict': model.state_dict()},
                    save_path)

    tb.add_hparams(
        {"lr": args.lr, "bsize": args.batch_size, "shuffle": args.shuffle},
        {
            # these are score at the end of training, probably overfit.
            "accuracy": train_acc,
            "loss": train_loss,
            # this is from early stop spot
            "best_accuracy": best_accuracy,
        },
    )

    tb.close()
    return

# def train(args, model, train_dataloader, val_dataloader, device='cpu'):
#     lr = args.lr
#     epochs = args.epochs
#     optimizer = Adam(model.parameters())
#     # optimizer = SGD(model.parameters(), lr=lr)
#     criterion = CrossEntropyLoss()
#     comment = f' batch_size = {args.batch_size} lr = {args.lr} shuffle = {args.shuffle} epochs = {args.epochs} cnn-lstm'
#     tb = SummaryWriter(comment=comment)

#     model.to(device)
#     for epoch in range(epochs):
#         # print("Epoch", epoch + 1)
#         model.train()
#         training_loss = 0.0

#         train_loss = 0.0
#         val_loss = 0.0
#         total_correct = 0.0
#         val_total_correct = 0.0
#         train_acc = 0.0
#         val_acc = 0.0
#         num_batches_used = 0.0 

#         for  data in train_dataloader:
#             # print("training on", i)
#             # if i % 30 == 0:

#             # get the inputs; data is a list of [inputs, labels]
#             # print(data.shape)
#             inputs, labels = get_augmented_batch(data)
#             # print(labels)
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs.float()  # shouldn't stay on this step.
#             # print("input shape", inputs.shape)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # print(inputs.shape)
#             outputs, _ = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             training_loss += loss.item()

#         model.eval()
#         valid_loss = 0.0
#         for inputs, labels in val_dataloader:
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs.float()  # shouldn't stay on this step.

#             logits, hidden_state = model(inputs)
#             # print(logits.shape)
#             _, pred = torch.max(logits, 1)
#             loss = criterion(logits, labels)
#             valid_loss += loss.item()

#         print(f'Epoch {epoch+1} \t\t Training Loss: {training_loss / len(train_dataloader) }\
#              \t\t Validation Loss: {valid_loss / len(val_dataloader )}')

#         # TODO - stop training when Val drops? val score is wack right now

#     if args.save:
#         print(args.save)
#         print("Saving model")
#         torch.save(model.state_dict(), args.save)

#     return


def test(args,  model, test_dataloader, show_plots=True, device='cpu'):
    model.eval()  # is necessary?

    with torch.no_grad():
        y_true = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            # calculate outputs by running images through the network
            logits, _ = model(images)
            # print(logits.shape)
            _, predicted = torch.max(logits, 1)
            # print(predicted.shape)
            # bad approach?
            y_pred = torch.cat((y_pred, predicted), 0)
            y_true = torch.cat((y_true, labels), 0)

    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())

    assert len(y_pred) == len(y_true)
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print("Accuracy:", accuracy)

    if show_plots:
        print(args.classes)
        print(cm)  # TODO - make pretty

    return


def get_augmented_batch(data):
    """
    applying same transform to each image was ugly. This function can never see the light of day. 
    """
    inputs, labels = data
    assert args.sequence == 10;
    new_images = inputs.clone().detach()
    new_labels = labels.clone().detach()
    A_transforms = [
        # [A.Sharpen(alpha=(.5, 1.), lightness=(0.1, 0.1), always_apply=True), ToTensorV2()],
        [A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=True), ToTensorV2()],
        [A.Superpixels(p_replace=0.1, n_segments=200, max_size=128, interpolation=1, always_apply=True), ToTensorV2()],
        # [A.Transpose(p=1), ToTensorV2()],
        [A.Blur(blur_limit=7, always_apply=True), ToTensorV2()],
        [A.RandomRotate90(p=1.0), ToTensorV2()],
        # [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=1.0), ToTensorV2()]
    ]

    for t in A_transforms:
        t = A.Compose(t, additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image', 'image8': 'image', 'image9': 'image'})
        # extend labels by one original batch
        new_labels = torch.cat((new_labels, labels))

        for sample in inputs:
            # aug = [t(image=channel.numpy())["image"]
            #        for frame in sample
            #        for channel in frame]
            # # print(len(aug))
            # # print(aug[0].shape)
            # aug = torch.stack(aug, dim=1)
            # aug = aug.unsqueeze(2)
            # print("###")


            # print(sample.shape)
        
            transformed = t(image=sample[0][0].numpy(), image1=sample[1][0].numpy(), image2=sample[2][0].numpy(), image3=sample[3][0].numpy(), image4=sample[4][0].numpy(), image5=sample[5][0].numpy(), image6=sample[6][0].numpy(), image7=sample[7][0].numpy(), image8=sample[8][0].numpy(), image9=sample[9][0].numpy())
            # print(transformed["image"].shape)
            aug = torch.stack((transformed["image"], transformed["image1"], transformed["image2"], transformed["image3"], transformed["image4"], transformed["image5"], transformed["image6"], transformed["image7"], transformed["image8"], transformed["image9"]), dim=0)
            aug = aug.unsqueeze(0)

            # print(new_images.shape, aug.shape)
            new_images = torch.cat((new_images, aug))
            # print(new_images.shape)

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
        inputs.size(0) + inputs.size(0)

    return new_images, new_labels


    """
    
    for t in A_transforms:
        t = A.Compose(t, additional_targets={'image1': 'image'})
        # extend labels by one original batch
        new_labels = torch.cat((new_labels, labels))

        for sample in inputs:
            transformed = t(image=sample[0].numpy(), image1=sample[1].numpy())
            aug = torch.stack((transformed["image"], transformed["image1"]), dim=1)
            new_images = torch.cat((new_images, aug))

    # make sure general shape of data is correct
    assert inputs[0].shape == new_images[0].shape
    assert new_labels.size(0) == len(A_transforms) * \
        inputs.size(0) + inputs.size(0)

    return new_images, new_labels
    """


def get_dataloaders(args, accept_list,  resize):
    """
    Access ornet dataset, apply any necessary transformations to images, 
    split into train/test/validate, and return dataloaders
    """

    #
    if args.roi:
        print("Using ROI inputs")
        transform = transforms.Compose([RoiTransform(window_size=(28, 28))])
    else:
        print("Using global image inputs")
        transform = transforms.Compose([transforms.Resize(size=resize)])
    dataset = DynamicVids(
        args.input_dir, accept_list, num_to_sample=args.sequence, class_types=args.classes, transform=transform) 

    print("dataset", len(dataset))

    # don't think this is the way to do this... needs even percent split to work
    train_split = .8
    train_size = math.floor(train_split * len(dataset))
    test_size = math.ceil((len(dataset) - train_size) / 2)
    val_size = math.floor((len(dataset) - train_size) / 2)
    train_size, test_size, val_size = int(
        train_size), int(test_size), int(val_size)

    assert train_size + val_size + test_size == len(dataset)
    # print("Train_size", train_size)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(69))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(73))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.weighted_samples:
        sampler = get_weighted_sampler(train_dataset, dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size)

    # print("Train_size", len(train_dataloader))
    return train_dataloader, test_dataloader, val_dataloader


def get_weighted_sampler(train_dataset, dataset):
    """
    Weighted Random Sampling
    One way to fight class imbalance
    """
    indices = train_dataset.indices
    y_train = [dataset.targets[i] for i in indices]  # Messed up this
    train_dataset_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / torch.tensor(train_dataset_count, dtype=torch.float)
    weights = weights[y_train]

    sampler = WeightedRandomSampler(weights, len(train_dataset))

    return sampler


if __name__ == "__main__":
    print("Main")

    # going to use the same parser manager for cnn and rnn
    args, _ = make_parser()

    # num_lstm_layers = 10  # too many?
    # lstm_hidden_size = -1 # if we decide to have a different hidden space size then must add param to model


    device = 'cpu' if args.cuda == 0 or not torch.cuda.is_available() else 'cuda'


    # device = 'cpu'
    print("device", device)



    """
    we have more segmented cell videos saved on logan then intermediates.
    best to rerun ornet on raw data, but till discrepancy is resolved this
    will result in using the 114 samples that Neelima used in the last scipy submit
    class balance 29, 31, 54
    """
    path_to_intermediates = "/data/ornet/gmm_intermediates"
    accept_list = []
    for subdir in args.classes:
        path = os.path.join(path_to_intermediates, subdir)
        for file in os.listdir(path):
            if 'normalized' in file:
                accept_list.append(file.split(".")[0])

    # train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args, accept_list)
    # print("Train_size", len(train_dataloader))


    hyper_parameters = dict(
        # lr=[0.001, 0.0001, 0.00001],
        lr=[0.0001],
        # lr=[0.0001, 0.00001],
        # batch_size=[16, 32, 64],
        batch_size=[32],
        # shuffle=[True, False]
        shuffle=[True]
    )
    param_values = [v for v in hyper_parameters.values()]

    for lr,batch_size, shuffle in product(*param_values):

        print(lr, batch_size, shuffle)
        args.lr = lr
        args.batch_size = batch_size
        args.shuffle = shuffle

        model = CNN_LSTM(args.sequence)

        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
            args, accept_list, resize=28)

        model.to(device)
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)

    
    if args.train:
        print("Training")
        train(args, model, train_dataloader, val_dataloader, device=device)

    if args.test:
        checkpoint = torch.load(args.save_model)
        model.load_state_dict(checkpoint['state_dict'])
        print("Testing")

        test(args, model, test_dataloader, device=device)

    # get features from final model.
    # if args.save_features:
    #     checkpoint = torch.load(args.save_model)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("Getting deep features")
    #     loader_dict = {
    #         "train": train_dataloader, "test": test_dataloader, "val": val_dataloader
    #     }
    #     feature_dict = get_deep_features(args, model, loader_dict, device=device)
    #     # with open(args.save_features, 'wb') as f:
    #     #     pickle.dump(feature_dict, f)
