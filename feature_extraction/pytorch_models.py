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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec=torchaudio.functional.spectrogram
SR=44100


def env_Model(D_in=40,H1=10,H2=5,H3=3,H4=2,H5=10,D_out=2,device="cpu"):
        D_in=D_in
        
        model_env = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.PReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.PReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.PReLU(),
        torch.nn.Linear(H3, D_out),
        torch.nn.Softmax())
        
        model_env.to(device)
        return model_env
    
class env_LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=x_features,    
                hidden_size=hidden_size,  
            ),
            'linear': nn.Linear(
                in_features=hidden_size,
                out_features=1)
        })

    def forward(self, x):

        # From [batches, seqs, seq len, features]
        # to [seq len, batch data, features]
        x = x.view(x_seq_len, -1, x_features)
        # Data is fed to the LSTM
        out, _ = self.model['lstm'](x)
        print(f'lstm output={out.size()}')

        # From [seq len, batch, num_directions * hidden_size]
        # to [batches, seqs, seq_len,prediction]
        out = out.view(x_batches, x_seqs, x_seq_len, -1)
        print(f'transformed output={out.size()}')

        # Data is fed to the Linear layer
        out = self.model['linear'](out)
        print(f'linear output={out.size()}')

        # The prediction utilizing the whole sequence is the last one
        y_pred = out[:, :, -1].unsqueeze(-1)
        print(f'y_pred={y_pred.size()}')

        return y_pred

def freq_model( D_in=50,H1=4,H2=2,H3=2,D_out=2,device="cpu"):
    freq_model = torch.nn.Sequential(
                        torch.nn.Linear(D_in, H1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(H1,H2),
                        torch.nn.PReLU(),
                        torch.nn.Linear(H2,H3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(H3, D_out),
                        torch.nn.Softmax())
    return freq_model

def env_freq_Model(D_in=50,H1=4,H2=2,H3=2,H4=10,H5=10,D_out=2,device="cpu"):
        D_in=D_in
        
        model_env = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.PReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.PReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.PReLU(),
        torch.nn.Linear(H3, D_out),
        torch.nn.Softmax())
        
        model_env.to(device)
        return model_env

def getFCSpecModel(D_in=400,H1=200,H2=50,H3=25,H4=10,H5=10,D_out=9):
        model_pitch = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.PReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.PReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.PReLU(),
        torch.nn.Linear(H3, D_out),
        torch.nn.Softmax())
        return model_pitch
    
class CNN_dvn(nn.Module):
    def __init__(self):
        super(CNN_dvn, self).__init__()
#         self.adapt = nn.AdaptiveMaxPool2d((20,20))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(20 * 20 * 8, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)
        self.lsm=torch.nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out=self.lsm(out)
        return out
    
class CNNLSTM_dvn(nn.Module):
    def __init__(self):
        super(CNNLSTM_dvn, self).__init__()
#         self.adapt = nn.AdaptiveMaxPool2d((20,20))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(7,3), stride=1, padding=(3,1)),
            nn.ReLU(),
        )
        self.drop_out = nn.Dropout()
        self.l1 = nn.LSTMCell(20 * 20 * 2, 20)
        self.fc3 = nn.Linear(20, 2)
        self.lsm=torch.nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
#         out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out,_ = self.l1(out)
#         out = self.fc2(out)
        out = self.fc3(out)
        out=self.lsm(out)
        return out

class CNNLSTM_dvd(nn.Module):
    def __init__(self,len_out=6):
        super(CNNLSTM_dvd, self).__init__()
        self.len_out=len_out
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(7,3), stride=1, padding=(3,1)),
            nn.ReLU(),
        )
        self.drop_out = nn.Dropout()
        self.l1 = nn.LSTMCell(20 * 20 * 2, 20)
        self.fc3 = nn.Linear(20, self.len_out)
        self.lsm=torch.nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out,_ = self.l1(out)
        out = self.fc3(out)
        out=self.lsm(out)
        return out

class CNN_dvd(nn.Module):
    def __init__(self,len_out=6):
        super(CNN_dvd, self).__init__()
        self.len_out=len_out
#         self.adapt = nn.AdaptiveMaxPool2d((20,20))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(20 * 20 * 8, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, self.len_out)
        self.lsm=torch.nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out=self.lsm(out)
        return out
    
    
class freqTrans(object):
    def __init__(self,num_mels=50,SR=SR):
        self.num_mels=num_mels
        self.ampT=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=30)
        self.melF=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=0.0, f_max=None, n_stft=None)
        
    def call(self, sample):
        wf,label=sample["signal"],sample["label"]
        wf=wf.reshape(-1,len(wf))
        sample_length=SR//4
#         wf=wf[:,0:24000]
        num_bins=wf[0].shape[0]
        win_length=num_bins
        hop_step=sample_length//(self.num_mels)
        window=torch.tensor([1]*win_length)
        s=spec(wf, 100, window, num_bins, hop_step, win_length,2,normalized=False)
        s=self.melF(s)
        s=self.ampT(s)
        freq=s.sum(axis=0).sum(axis=1)
        freq=freq/freq.abs().max()
        freq[torch.isnan(freq)]=0
        return {"feats":freq.detach(),"label":label}
    
class envTrans(object):

    def __init__(self,num_mels=1,SR=SR):
        self.env_size=9
        self.num_mels=num_mels
        self.amp=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=60)
        self.melEnv=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=30.0, f_max=None, n_stft=None)
#         self.norm= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    def call(self, sample):
        wf,label=sample["signal"],sample["label"]

        wf=wf.reshape(-1,len(wf))
        sample_length=SR
        num_bins=wf[0].shape[0]
        win_length=SR//10
        hop_step=SR//self.env_size
        window=torch.tensor([1]*win_length)
        s=spec(wf, 0, window,win_length, hop_step, win_length,2,normalized=False)
        s=self.melEnv(s)
        s=self.amp(s)
#         s=self.norm(s)
        #normalizing
        env=s.sum(axis=0).sum(axis=0)
        env=env-env.min()
        env=env/env.max()
        env[torch.isnan(env)]=0

        num_padding=torch.max(torch.tensor([self.env_size+1-env.shape[0],0]))
        env_vec=torch.cat([env.detach(),torch.zeros(num_padding)],dim=0)
        return {"feats":env_vec.detach(),"label":label}

class specTrans(object):
    def __init__(self,num_mels=50,SR=SR):
        self.num_mels=num_mels
        self.ampP=torchaudio.transforms.AmplitudeToDB(stype='power',top_db=40)
        self.melP=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=30.0, f_max=15000.0, n_stft=None)
        self.norm= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    def call(self, sample):
        
        wf,label=sample["signal"],sample["label"]
        wf=wf.reshape(-1,len(wf))
        sample_length=SR

        num_bins=wf[0].shape[0]
        win_length=SR//17
        hop_step=SR//19
        window=torch.tensor([1]*win_length)
        s=spec(wf, 0, window, num_bins, hop_step, win_length,2,normalized=False)
        s=self.melP(s)
        s=self.ampP(s)
        s=s/s.abs().max()
#         freq=self.norm(s)
        freq=s
        return {"feats":freq.detach(),"label":label}
    
class freq_and_env_Trans(object):
    def __init__(self,feat_mels=50,env_mels=1):
        self.et=envTrans(num_mels=env_mels)
        self.ft=freqTrans(num_mels=feat_mels)

    def call(self, sample):
            ftr=self.ft.call(sample)["feats"]
            etr=self.et.call(sample)["feats"]

            combined_feats=torch.cat((ftr,etr))
            return {"feats":combined_feats,"label":sample["label"]}
        

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

class AE_Linear(nn.Module):
    def __init__(self,compression_dim=64,decoder_dims=256,**kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.encoder_output_layer = nn.Linear(
            in_features=256, out_features=compression_dim
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=compression_dim, out_features=decoder_dims
        )
        self.decoder_output_layer = nn.Linear(
            in_features=decoder_dims, out_features=kwargs["input_shape"]
        )
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        self.encoding=code
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed