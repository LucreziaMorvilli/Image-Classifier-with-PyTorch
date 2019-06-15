#Imports here
import numpy as np
import pandas as pd
import time

import functions

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

from collections import OrderedDict

import json

from workspace_utils import active_session

from sklearn.preprocessing import normalize

import argparse

#define parser
parser = argparse.ArgumentParser(description='Training the model')
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--device', dest="device", action="store", default="gpu")
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--model', dest="model", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=[1024,256])

#call parser and store variables
pa = parser.parse_args()
lr = pa.learning_rate
data_dir = pa.data_dir
save_checkpoint = pa.save_dir
device = pa.device
model = pa.model
epochs = pa.epochs
dropout = pa.dropout
hidden_units = pa.hidden_units

#check input
print("The input variables are:\n",pa)

#LOAD DATA
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

#Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Load pre-trained network
if model == 'vgg16':
    my_model = models.vgg16(pretrained=True)
elif model == 'densenet121':
    my_model = models.densenet121(pretrained=True)
elif model == 'alexnet':
    my_model = models.alexnet(pretrained = True)
else:
    print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(model))

print("\nPre-trained model has been downloaded")

#Freeze parameters to avoid backpropagation
for param in my_model.parameters():
    param.requires_grad = False
    
arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

# using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch[model], hidden_units[0])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units[0],hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_units[1], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
my_model.classifier = classifier
print("\nClassifier has been created")

# Train the classifier layers using backpropagation using the 
# pre-trained network to get the features

#define loss
criterion = nn.NLLLoss()

#optimizer (training parameters of classifier only)
optimizer = optim.Adam(my_model.classifier.parameters(), lr)

#depending on cpu/gpu usage
if torch.cuda.is_available() and device == 'gpu':
    device = 'cuda'
    my_model.to(device)

#TRAINING THE MODEL
train_losses = []
test_losses = []
running_loss = 0
steps = 0
print_every = 5

start = time.time()
print("\nStarting to train the model...")

with active_session():
    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            ## Training loop ##

            optimizer.zero_grad()
            logps = my_model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ## Validation Loop ##

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                my_model.eval()

                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    logps = my_model.forward(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                    #calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"\tEpoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                my_model.train()        
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(valid_loss/len(testloader))
               

    print(f"Total Time: {(time.time() - start):.3f} seconds") 
print('\nTraining done')

#SAVING CHECKPOINT
my_model.class_to_idx = train_data.class_to_idx 
checkpoint = {'model':model,
              'input_size': arch[model],
              'output_size': 102,
              'hidden_size': hidden_units,
              'state_dict': my_model.state_dict(),
              'mapping': my_model.class_to_idx,
              'optimizer': optimizer.state_dict,
              'epochs': epochs,
              'dropout': dropout
            }

torch.save(checkpoint, save_checkpoint)
print('\nSaved checkpoint')
