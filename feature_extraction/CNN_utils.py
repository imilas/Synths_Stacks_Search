import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_net(torch.nn.Module):
    #Our batch shape for input x is (3, 128, 128)
    def __init__(self):
        #### look up nn.sequentional  for conv+relu in place
        super(CNN_net, self).__init__()
        
        #Input channels = 1
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        self.fc1 = nn.Linear(8*60*60, 128)
        self.fc3 = torch.nn.Linear(128, 9)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8*60*60)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return(x)