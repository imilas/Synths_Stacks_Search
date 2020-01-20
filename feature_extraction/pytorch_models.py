import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torchaudio
import torchvision as tv

spec=torchaudio.functional.spectrogram
SR=48000

def env_Model(D_in=31,H1=4,H2=2,H3=2,H4=10,H5=10,device="cpu"):
        D_in,D_out =D_in,2
        
        model_env = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.PReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.PReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.PReLU(),
        torch.nn.Linear(H3,H4),
        torch.nn.PReLU(),
        torch.nn.Linear(H4, D_out),
        torch.nn.Softmax())
        model_env.to(device)
        return model_env

def freq_model(H1=4,H2=2,H3=2):
    BATCH_SIZE, D_in,D_out =2,50,2


    freq_model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H1),
                torch.nn.ReLU(),
                torch.nn.Linear(H1,H2),
                torch.nn.PReLU(),
                torch.nn.Linear(H2,H3),
                torch.nn.ReLU(),
                torch.nn.Linear(H3, D_out),
                torch.nn.Softmax()
            )
    return freq_model

def pitch_model(D_in=20*21,H1=400,H2=200,H3=100,H4=50,H5=10,D_out=2,device="cpu"):
#         D_in=20*21,H1=400,H2=200,H3=100,H4=50,H5=10
        
        model_pitch = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.PReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.PReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.PReLU(),
        torch.nn.Linear(H3,H4),
        torch.nn.PReLU(),
        torch.nn.Linear(H4, D_out),
        torch.nn.Softmax())
        
        model_pitch.to(device)
        return model_pitch

#defining transformations
class pitchTrans(object):
    def __init__(self,num_mels=20,SR=SR):
        self.num_mels=num_mels
        self.ampP=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=30)
        self.melP=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=0.0, f_max=None, n_stft=None)
        
    def call(self, sample):
        wf,label=sample["signal"],sample["label"]
        wf=wf.reshape(-1,len(wf))
        sample_length=SR//4
        wf=wf[:,0:24000]
        num_bins=wf[0].shape[0]
        win_length=400
        hop_step=599
        window=torch.tensor([1]*win_length)
        s=spec(wf, 10, window, num_bins, hop_step, win_length,2,normalized=False)
        s=self.melP(s)
        s=self.ampP(s)
        s=s/s.abs().max()
        freq=s
#         print(s.shape)

#         print(freq,freq.shape)

        return {"feats":freq.detach(),"label":label}
    
class freqTrans(object):
    def __init__(self,num_mels=50,SR=SR):
        self.num_mels=num_mels
        self.ampT=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=30)
        self.melF=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=0.0, f_max=None, n_stft=None)
        
    def call(self, sample):
        wf,label=sample["signal"],sample["label"]
        wf=wf.reshape(-1,len(wf))
        wf=wf[:,0:15000]
        num_bins=wf[0].shape[0]
        win_length=num_bins
        window=torch.tensor([1]*win_length)
        s=spec(wf, 100, window, num_bins, win_length, win_length,2,normalized=False)
        s=self.melF(s)
        s=self.ampT(s)
        freq=s.sum(axis=0).sum(axis=1)
        freq=freq/freq.abs().max()
        freq[torch.isnan(freq)]=0
        return {"feats":freq.detach(),"label":label}
    
class envTrans(object):
    def __init__(self,num_mels=10,SR=SR):
        self.num_mels=num_mels
        self.amp=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=30)
        self.melEnv=torchaudio.transforms.MelScale(n_mels=2*self.num_mels, sample_rate=SR, f_min=0.0, f_max=None, n_stft=None)
    def call(self, sample):
        wf,label=sample["signal"],sample["label"]

        wf=wf.reshape(-1,len(wf))
        wf=wf[:,0:15000]
        num_bins=wf[0].shape[0]//10
        win_length=num_bins
        window=torch.tensor([1]*win_length)
        s=spec(wf, 0, window, num_bins, win_length, win_length,2,normalized=False)
        s=self.melEnv(s)

        env=s.sum(axis=0).sum(axis=0)
        env=env/env.abs().max()
        env[torch.isnan(env)]=0

        num_padding=torch.max(torch.tensor([self.num_mels+1-env.shape[0],0]))
        env_vec=torch.cat([env.detach(),torch.zeros(num_padding)],dim=0)
        return {"feats":env_vec.detach(),"label":label}
        

class LSTM(nn.Module):
    def __init__(self, input_dim,seq_dim,hidden_dim,n_layers,output_size,device="cpu"):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        self.seq_dim=seq_dim
        self.input_dim=input_dim
        self.device=device
    def forward(self, x, hidden="what"):
        x=x.view(-1, self.seq_dim, self.input_dim).requires_grad_()
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    def init_hidden(self, batch_size,device="NotGiven"):
        if device=="NotGiven":
            device=self.device
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden   
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(25 * 30 * 64, 1000)
        self.fc2 = nn.Linear(1000, 7)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
# # transformation
# def t(z):
#     #normalize array->pilform ->apply transoforms,
#     z=(((z - z.min()) / (z.max() - z.min())) * 254).astype(np.uint8)
#     zi=Image.fromarray(z)
#     z=t(zi)
#     return z

# t = transforms.Compose(
#     [transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(), 
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
