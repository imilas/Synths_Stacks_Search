# %reload_ext autoreload
# %autoreload 1   
import numpy as np
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import pytorch_models
import imp
import torch.optim as optim

import matplotlib.pyplot as plt
import time
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
EPOCH_SIZE = 512
TEST_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.01

TRAIN_DATA_PATH = "/home/asalimi/Synths_Stacks_Search/feature_extraction/lstm_data/simple/train"
TEST_DATA_PATH = "/home/asalimi/Synths_Stacks_Search/feature_extraction/lstm_data/simple/test"
TRANSFORM_IMG = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def get_loaders(bs=BATCH_SIZE):
    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_loader = data.DataLoader(train_data, batch_size=bs,shuffle=True, num_workers=4)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_loader  = data.DataLoader(test_data, batch_size=bs, shuffle=True, num_workers=4)
    val_loader=test_loader
    return train_loader,test_loader,train_data,test_data,val_loader

train_loader,test_loader,train_data,test_data,val_loader = get_loaders()

classes=list(train_data.class_to_idx.keys()) #get list of classes

def train(model, optimizer, train_loader, device=torch.device("cpu")):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        h = model.init_hidden(data.size()[0])
        output,h = model(images,h)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            h = model.init_hidden(data.size()[0])
            output,h = model(images,h)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

def train_lstm(config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    while True:
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)
        track.log(mean_accuracy=acc)
        
class LSTM(nn.Module):
    def __init__(self, input_dim,hidden_dim, n_layers,output_size):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        x=x.view(-1, seq_dim, input_dim).requires_grad_()
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
        
input_dim = 120
seq_dim=100
output_size = 5
hidden_dim = 1000
n_layers = 1
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)  

def train_LSTM(config):
    train_loader,test_loader,train_data,test_data,val_loader = get_loaders(bs=config["bs"])
    model = LSTM(input_dim, hidden_dim, n_layers,output_size,)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        tune.track.log(mean_accuracy=acc)
analysis = tune.run(
    train_LSTM, config={"lr": tune.grid_search([0.001, 0.01, 0.1]),"bs": tune.grid_search([2,4,8,16])})

print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()