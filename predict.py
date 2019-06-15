#Imports 
import numpy as np
import pandas as pd
import time

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
import functions

#define parser
parser = argparse.ArgumentParser(description='Making predictions')
parser.add_argument('checkpoint', default="./checkpoint.pth", nargs='*', action="store", type = str)
parser.add_argument('--device', dest="device", action="store", default="gpu")
parser.add_argument('--input_image', default="./flowers/test/1/image_06743.jpg",nargs = '*',action = 'store', type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

#call parser and store variables
pa = parser.parse_args()
checkpoint = pa.checkpoint
device = pa.device
input_image = pa.input_image
top_k = pa.top_k
category_names = pa.category_names

#check input
print("The input variables are:\n",pa)

#LOAD CHECKPOINT
my_model, optimizer = functions.load_checkpoint(checkpoint)
print('\nCheckpoint Loaded')

#PROCESS IMAGE
test= functions.process_image(input_image)
print("\nImage processed")

#depending on cpu/gpu usage
if torch.cuda.is_available() and device == 'gpu':
    device = 'cuda'
    my_model.to(device)

#PREDICT CATEGORY NAME
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, classes, names = functions.predict(test, my_model, device, top_k, cat_to_name)
print('\nPredictions made. Here they are, together with their corresponding probability:\n')
#result=pd.DataFrame(probs,names)
result = pd.DataFrame(probs.data.to('cpu').numpy().squeeze(),names)
result.columns = ['Probability']
print(result)



