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

def make_configs():
    print("Building Configurations Now...")
    CONF_LIST = []
    all_configs = itertools.product(LR, BATCHES, OPTIMS, MOMENT, WEIGHT_DECAY)
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

def train_model(config, test_loader, criterion, task='Classification', numEpochs=1):
    print("Training :{}".format(config.name))
    config.model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(tqdm(config.dataloader)):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            
            config.optimizer.zero_grad()
            outputs = config.model(feats)

            loss = criterion(outputs, labels.long())
            loss.backward()
            config.optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 500 == 499:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/500))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = test_classify(config.model, test_loader, criterion)
            train_loss, train_acc = test_classify(config.model, config.dataloader, criterion)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)


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


def test_verify(model, test_loader):
    raise NotImplementedError