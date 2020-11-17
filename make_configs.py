import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import DenseNet121
import torchvision

from collections import namedtuple
import itertools
from tqdm import tqdm

HP_Config = namedtuple('HP_Config', 'name model dataloader optimizer')

lr = [1e-1, 1e-2, 1e-3]
batches = [32, 64, 128]
optims = ['Adam', 'SGD', 'RMSprop']
moment = [0.7, 0.8, 0.9]
weight_decay = [0.7, 0.8, 0.9]
all_configs = itertools.product(lr, batches, optims, moment, weight_decay)
NUM_CLASSES = 100

#Set torch global veriable
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'

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
for bs in batches:
    DATALOADER_DICT[bs] = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8 if CUDA else 1)

def make_configs(all_configs=all_configs):
    print("Building Configurations Now...")
    CONF_LIST = []
    for idx, conf in enumerate(tqdm(list(all_configs))):
        lr = conf[0]
        bs = conf[1]
        opt = conf[2]
        mm = conf[3]
        wd = conf[4]
        conf_name = 'conf_{}_{}_{}_{}_{}'.format(lr, bs, opt, mm, wd)
        model = DenseNet121()
        dataloader = DATALOADER_DICT[bs]
        optimiser = None
        # Make optimiser
        if opt == 'Adam':
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt == 'SGD':
            optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mm)
        elif opt == 'RMSprop':
            optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=mm)

        CONF_LIST.append(HP_Config(conf_name, model, dataloader, optimiser))

    return CONF_LIST