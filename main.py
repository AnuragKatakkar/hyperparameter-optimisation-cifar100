# Get Configurations and Perform Your Choice of HP Optimisation here
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

from tqdm import tqdm
from utils import train_model, make_configs

#Globals
CUDA = torch.cuda.is_available
NUM_WORKERS = 8 if CUDA else 1
DEVICE = 'cuda' if CUDA else 'cpu'
BATCH_SIZE = 128
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

#Get all Configurations as a list
conf_list = make_configs()

#Implement Random Search/Bayesian Opt/Hyperband here
conf_list[0].model.train()
conf_list[0].model.to(DEVICE)
train_model(conf_list[0], testloader, criterion)