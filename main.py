import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

import numpy as np
from tqdm import tqdm
from utils import train_model, get_config, plot_val_loss
from numpy import argsort
from math import log, ceil
import random
import matplotlib.pyplot as plt

#Globals
CUDA = torch.cuda.is_available
NUM_WORKERS = 8 if CUDA else 1
DEVICE = 'cuda' if CUDA else 'cpu'
BATCH_SIZE = 32
BATCHES = [32, 64, 128]
criterion = torch.nn.CrossEntropyLoss()

#Make test dataset and dataloader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Prep dataloaders
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# Create the necessary Dataloaders here
DATALOADER_DICT = {}
for bs in BATCHES:
    DATALOADER_DICT[bs] = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8 if CUDA else 1)

#Implement your HP Optimisation Technique Here:
# This implementation of Hyperband is taken from the website : https://docs.google.com/document/d/1Wp8_DiIlnAb6b_GTkZ0dWWjq6nbL4oP09wI6CBtu1CY/edit
# And is authored by Kevin Jamieson

max_iter = 81  # maximum iterations/epochs per configuration
eta = 3 # defines downsampling rate (default=3)
logeta = lambda x: log(x)/log(eta)
s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

ALL_MODELS = {s:None for s in range(s_max + 1)}
#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max+1)):
    print("Running s=", s)
    n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    T = [ get_config() for i in range(n) ]
    ALL_MODELS[s] = T
    for i in range(s+1):
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        n_i = n*eta**(-i)
        r_i = r*eta**(i)
        print("Running n_i={}, r_i={}".format(n_i, r_i))
        val_losses = []
        for t in T:
          if t.dataloader == None:
            t.set_dataloader(DATALOADER_DICT[t.bs])
          t.model.train()
          t.model.to(DEVICE)
          val_losses.append(train_model(t, testloader, criterion, numEpochs=int(r_i)))
        T = [ T[i] for i in argsort(val_losses)[0:int( n_i/eta )] ]
    val_losses = [conf.val_losses for conf in ALL_MODELS[s]]
    plot_val_loss(val_losses, s)
    T[0].best = True
    #### End Finite Horizon Successive Halving with (n,r)

