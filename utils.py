import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import DenseNet121
import torchvision
import torch.nn.functional as F

from collections import namedtuple
import itertools
from tqdm import tqdm
import numpy as np
import random

HP_Config = namedtuple('HP_Config', 'name model dataloader optimizer')

LR = [1e-1, 1e-2, 1e-3]
BATCHES = [32, 64, 128]
OPTIMS = ['Adam', 'SGD', 'RMSprop']
MOMENT = [0.8, 0.85, 0.9]
WEIGHT_DECAY = [0.7, 0.8, 0.9]
NUM_CLASSES = 100
CUDA = torch.cuda.is_available
DEVICE = 'cuda' if CUDA else 'cpu'

#Set torch global veriable
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'

class HP_Config:
    def __init__(self, bs, lr, opt, opt_name, mm, wd, model):
        self.lr = lr
        self.bs = bs
        self.optimizer = opt
        self.opt_name = opt_name
        self.mm = mm
        self.wd = wd
        self.name = 'conf_{}_{}_{}_{}_{}'.format(lr, bs, opt_name, mm, wd)
        self.val_losses = []
        self.val_acc = []
        self.model = model
        self.dataloader = None
        self.epochs_run = 0
        self.best = False
    
    def set_dataloader(self, dataloader):
      self.dataloader = dataloader

def get_config(lr=None, bs=None, opt=None, mm=None, wd=None):
    if lr == None:
        lr = np.random.normal(1e-3 ,1e-4)
        while (lr<0):
            lr = np.random.normal(1e-3, 1e-4)
    if bs == None:
        bs = random.choice(BATCHES)
    if opt == None:
        opt = random.choice(OPTIMS)
    if mm == None:
        mm = np.random.normal(0.9, 1e-3)
    if wd == None:
        wd = np.random.normal(5e-6, 1e-7)
    model = DenseNet121()
    optimiser = None
    # Make optimiser
    if opt == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mm)
    elif opt == 'RMSprop':
        optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=mm)

    return HP_Config(bs, lr, optimiser, opt, mm, wd, model)

def train_model(config, test_loader, criterion, task='Classification', numEpochs=1):
    print("Training :{}".format(config.name))
    config.model.train()

    for epoch in range(numEpochs):
        config.epochs_run += 1
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(tqdm(config.dataloader)):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            
            config.optimizer.zero_grad()
            outputs = config.model(feats)

            loss = criterion(outputs, labels.long())
            loss.backward()
            config.optimizer.step()
            
            avg_loss += loss.item()

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        val_loss, val_acc = test_classify(config.model, test_loader, criterion)
        config.val_losses.append(val_loss)
        config.val_acc.append(val_acc)
        print("Epoch : {}, Avg Train Loss : {}, Val Loss : {}, Val Acc : {}".format(config.epochs_run, avg_loss/len(config.dataloader), val_loss, val_acc))
    return val_loss

def test_classify(model, test_loader, criterion):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


def plot_val_loss(val_losses, s):
  colors = ['b', 'k', 'r', 'c', 'o', 'g', 'y']
  plt.figure()
  for vl in val_losses:
    plt.plot(np.arange(0, len(vl), 1), vl, random.choice(colors))

  plt.savefig('HP_s_{}'.format(s))