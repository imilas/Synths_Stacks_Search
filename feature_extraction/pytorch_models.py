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

class audioDataset(torch.utils.data.Dataset):
    def __init__(self,audio_frame,root_dir, task="keep_all",transform=None):
        self.root_dir=root_dir
        self.task=task
        self.audio_frame=audio_frame
        self.transform = transform
        self.minLength=SR
#         self.minLength=SR//4
        self.frame_pruning()
    def __len__(self):
        return len(self.audio_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rows=self.audio_frame.iloc[idx]

        signals,labels=rows["audio"].tolist()[0:SR],rows["label_num"].tolist()
        signals,labels=torch.tensor(signals),torch.tensor(labels)
        
        nz=np.max((self.minLength-signals.shape[0],0))
        signals=torch.cat([signals[0:self.minLength],torch.zeros(nz)],dim=0)

        sound={"signal":signals,"label":labels,"path":rows["path"],"drum_type":rows["label"]}
        
        if self.transform:
            sound = self.transform(sound)

        return sound
    
    def frame_pruning(self):
        #drum vs not drum classification:
        if self.task=="dvn":
            self.audio_frame.loc[self.audio_frame["label"]!="synth_noise","label_num"]=0 
            self.audio_frame.loc[self.audio_frame["label"]=="synth_noise","label_num"]=1 
        #drum type classification
        if self.task=="dvd":
            self.audio_frame=self.audio_frame.loc[self.audio_frame["label"]!="synth_noise"]
        if self.task=="keep_all":
            pass
        
        
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
    def __init__(self,len_out=5):
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

# auto encoder stuff
class AE_Conv5x5(nn.Module):
    def __init__(self,input_shape,compression_dim,dropout_rate=0.5,num_channels=5,eval_mode=False):
        super(AE_Conv, self).__init__()
        self.W=input_shape[0]
        self.H=input_shape[1]
        self.eval_mode = eval_mode
        self.dropout = nn.Dropout(dropout_rate)
        self.Encoder_Conv= nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=num_channels, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder_output_layer = nn.Linear(
            in_features=(self.H//2 * self.W//2) * 8, out_features=compression_dim
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=compression_dim, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=self.W*self.H)
        
        
    def forward(self, features):
        features=features.reshape([-1,1,self.W,self.H])
        activation = self.Encoder_Conv(features)
        activation = self.dropout(torch.relu(activation))
        activation = activation.reshape(activation.size(0), -1)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        self.encoding=code
        if self.eval_mode:
            return code
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
#optimized params
# TIME_STEPS = 20
# FREQ_BINS = 30
# l2 = 3.2473701348597023e-06
# #hyper params
# latent_size = 16
# amp_to_power=True
# num_channels=3
# spec_dimension=FREQ_BINS*TIME_STEPS
# learning_rate = 0.0011451089315356296
# dropout_rate = 0.5

class AE_Conv1x3(nn.Module):
    def __init__(self,input_shape,compression_dim,dropout_rate,num_channels=5,eval_mode=False):
        super(AE_Conv1x3, self).__init__()
        self.H=input_shape[0]
        self.W=input_shape[1]
        self.eval_mode=eval_mode
        self.dropout = nn.Dropout(dropout_rate)
        self.Encoder_Conv= nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=[1,3], stride=1, padding=1),
            nn.ReLU(),
          nn.MaxPool2d(kernel_size=[1,2], stride=[1,2]))
  
        self.encoder_output_layer = nn.Linear(
            in_features=(33 * (self.W//2)) * num_channels, out_features=compression_dim
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=compression_dim, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=self.W*self.H)
        
    def forward(self, features):
        features=features.reshape([-1,1,self.W,self.H])
        activation = self.Encoder_Conv(features)
        activation = self.dropout(torch.relu(activation))
        activation = activation.reshape(activation.size(0), -1)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        self.encoding=code
        if self.eval_mode:
            return code
        activation = self.decoder_hidden_layer(code)
        activation = self.dropout(torch.relu(activation))
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed    
class AE_envTrans(object):
    def __init__(self,num_mels=10,SR=SR):
        self.env_size=9
        self.num_mels=num_mels
        self.amp=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=60)
        self.melEnv=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR, f_min=10.0, f_max=None, n_stft=None)
#         self.norm= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    def __call__(self, sample):
        wf,label=sample["signal"],sample["label"]
        
        wf=wf.reshape(-1,len(wf))
        sample_length=SR
        num_bins=wf[0].shape[0]
        win_length=SR//20
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
    
class AE_specTrans(object):
    def __init__(self,num_mels=50,SR=SR,time_steps=20,amp_to_power=False):
        self.amp_to_power=amp_to_power
        self.num_mels=num_mels
        self.ampP=torchaudio.transforms.AmplitudeToDB(stype='power',top_db=10)
        self.melP=torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=SR,n_stft=None)
#         self.norm= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.hop_step=time_steps-1
    def __call__(self, sample):
        
        wf,label,p,drum_type=sample["signal"],sample["label"],sample["path"],sample["drum_type"]
        wf=wf.reshape(-1,len(wf))
        sample_length=SR

        num_bins=wf[0].shape[0]
        win_length=SR//17
        hop_step=SR//self.hop_step
        window=torch.tensor([1]*win_length)
        s=spec(wf, 0, window, num_bins, hop_step, win_length,2,normalized=False)
        s=self.melP(s)
        if self.amp_to_power:
            s=self.ampP(s)
        s = s - s.min()
        s = s/s.abs().max()

        freq=s
#         freq=self.norm(s)
        freq[torch.isnan(freq)]=0
        freq=freq[0]
        return {"feats":freq.detach(),"label":label,"path":p,"drum_type":drum_type}
    
#when i need both spec and env feats
class spec_and_env(object):
    def __init__(self,specTrans=None,envTrans=None):
        self.sT=specTrans
        self.eT=envTrans
    def __call__(self, sample):
            #will get meta data from spec
            return {"spec_trans_results":self.sT(sample),"env_trans_results":self.eT(sample)["feats"]}

# 1D conv signal classifier
class ConvSig_Classifier_DVN(nn.Module):
    def __init__(self,embed_only=False,dropout=0.05,device=device):
        super(ConvSig_Classifier_DVN, self).__init__()
        self.embed_only = embed_only
        self.dropout = dropout
        self.device = device
        self.l1 = nn.Sequential(
                nn.Conv1d(1,128,500, stride=2, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 250, stride=2, padding=4),
#                 nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 100, stride=2, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 100, stride=2, padding=2),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, 80, stride=1, padding=3),
#                 nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 40, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.l2 = nn.Sequential(
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128,128),
                nn.Dropout(self.dropout),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l3 = nn.Sequential(
                  nn.Linear(64,32),
                  nn.ReLU(),
                  nn.Dropout(self.dropout),
                  nn.Linear(32,16),
                  nn.ReLU(),
                  nn.Linear(16,2),
                )

    def forward(self, x_sig):
        
        x_sig = x_sig.float()
        bs = x_sig.shape[0]
        x_sig = x_sig.reshape(bs,1,-1).to(self.device)
        x1 = self.l1(x_sig)
        x1 = x1.reshape(bs,-1)
        x2 = self.l2(x1)
        x_agg = torch.cat((x2,), dim=1)
        x_final = self.l3(x_agg)
        return x_final

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
class Conv_Spec_DVN(nn.Module):
    def __init__(self,embed_only=False,dropout=0.025,device=device):
        super(Conv_Spec_DVN, self).__init__()
        self.embed_only = embed_only
        self.dropout = dropout
        self.device=device
        self.conv_1d = nn.Sequential(
                nn.Conv1d(1,128,500, stride=2, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 250, stride=2, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 100, stride=2, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 100, stride=2, padding=2),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, 80, stride=1, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 40, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.spectrogram_layer = nn.Sequential(
                nn.Linear(30*9,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64,32),
                nn.Dropout(self.dropout),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,16),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l2 = nn.Sequential(
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64,32),
                nn.Dropout(self.dropout),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l3 = nn.Sequential(
                  nn.Linear(16+16,32),
                  nn.ReLU(),
                  nn.Dropout(self.dropout),
                  nn.Linear(32,16),
                  nn.ReLU(),
                  nn.Linear(16,2),
                )

    def forward(self, x_sig,x_spec):
        x_sig = x_sig.float()
        bs = x_sig.shape[0]
        bs_spec = x_spec.shape[0]
        x_sig = x_sig.reshape(bs,1,-1).to(self.device)
        x1_1d = self.conv_1d(x_sig)
        flat_spec = x_spec.reshape([bs_spec,-1])
        x1_fc = self.spectrogram_layer(flat_spec)
        x1_1d = x1_1d.reshape(bs,-1)
        x2 = self.l2(x1_1d)
        x_agg = torch.cat((x2,x1_fc),dim=1)
        x_final = self.l3(x_agg)
        return x_final

        
class ConvSig_Classifier_DVD(nn.Module):
    def __init__(self,embed_only=False,dropout=0.1,device=device):
        super(ConvSig_Classifier_DVD, self).__init__()
        self.embed_only = embed_only
        self.dropout = dropout
        self.device=device
        self.l1 = nn.Sequential(
                nn.Conv1d(1,128,500, stride=2, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 250, stride=2, padding=4),
#                 nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 100, stride=2, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 100, stride=2, padding=2),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, 80, stride=1, padding=3),
#                 nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 40, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.l2 = nn.Sequential(
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128,128),
                nn.Dropout(self.dropout),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l3 = nn.Sequential(
                  nn.Linear(64,32),
                  nn.ReLU(),
                  nn.Dropout(self.dropout),
                  nn.Linear(32,16),
                  nn.ReLU(),
                  nn.Linear(16,5),
                )
    def forward(self, x_sig):
        
        x_sig = x_sig.float()
        bs = x_sig.shape[0]
        x_sig = x_sig.reshape(bs,1,-1).to(self.device)
        x1 = self.l1(x_sig)
        x1 = x1.reshape(bs,-1)
        x2 = self.l2(x1)
        x_agg = torch.cat((x2,), dim=1)
        x_final = self.l3(x_agg)
        return x_final

class Conv_Spec_DVD(nn.Module):
    def __init__(self,embed_only=False,dropout=0.075,device=device):
        super(Conv_Spec_DVD, self).__init__()
        self.embed_only = embed_only
        self.dropout = dropout
        self.device = device
        self.conv_1d = nn.Sequential(
                nn.Conv1d(1,128,500, stride=2, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 250, stride=2, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 100, stride=2, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 100, stride=2, padding=2),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, 80, stride=1, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 40, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.spectrogram_layer = nn.Sequential(
                nn.Linear(30*9,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64,32),
                nn.Dropout(self.dropout),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,16),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l2 = nn.Sequential(
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64,32),
                nn.Dropout(self.dropout),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                )
        self.l3 = nn.Sequential(
                  nn.Linear(16+16,32),
                  nn.ReLU(),
                  nn.Dropout(self.dropout),
                  nn.Linear(32,16),
                  nn.ReLU(),
                  nn.Linear(16,4),
                )

    def forward(self, x_sig,x_spec):
        x_sig = x_sig.float()
        bs = x_sig.shape[0]
        bs_spec = x_spec.shape[0]
        x_sig = x_sig.reshape(bs,1,-1).to(self.device)
        x1_1d = self.conv_1d(x_sig)
        flat_spec = x_spec.reshape([bs_spec,-1])
        x1_fc = self.spectrogram_layer(flat_spec)
        x1_1d = x1_1d.reshape(bs,-1)
        x2 = self.l2(x1_1d)
        x_agg = torch.cat((x2,x1_fc),dim=1)
        x_final = self.l3(x_agg)
        return x_final

    
    

import torchmetrics
from torchmetrics.functional import auc
from torch.nn import functional as F
from torch import nn
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import math
import torchaudio


import torchmetrics
from torchmetrics.functional import auc
from torch.nn import functional as F
from torch import nn
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import torchaudio
class Transformer_DVN(LightningModule):
    def __init__(self,attention_dropout=0.3,d_model=120,heads=30,encoding_layers=12,pool_dim=2,pct_start=0.05,max_lr=1e-4,max_momentum=0.95,epochs=50):
        super().__init__()
        dropout=0.2
        self.attention_dropout=attention_dropout
        self.d_model = d_model
        self.heads = heads
        self.encoding_layers = encoding_layers
        self.pool_dim = pool_dim
        self.pct_start = pct_start
        self.max_lr = max_lr
        self.max_momentum = max_momentum
        self.epochs = epochs
        self.spectrogram_func = torchaudio.transforms.Spectrogram(n_fft = int(self.d_model*2)-1, hop_length = 200, power = 0.2, normalized = True)
        
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.heads,
                                                        dropout = self.attention_dropout,)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoding_layers)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(self.pool_dim)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(self.pool_dim)
        
        self.decoder = nn.Sequential(
          nn.Linear(self.d_model*2*self.pool_dim,64),
          nn.Linear(64,32),
          nn.Linear(32,1),
        )
#         d["signal"]
    def forward(self, x):
        x = x.float()
        x1 = self.spectrogram_func(x).transpose(0,1).transpose(0,2)
        x1 = self.pos_encoder(x1)
        x2 = self.transformer_encoder(x1).transpose(1,0).transpose(2,1)
        x3r = self.adaptiveavgpool(x2)
        x3c = self.adaptivemaxpool(x2)
        x4 = torch.cat((x3r, x3c), dim=1)
        x4 = x4.view(x4.size(0), -1)
        out =  self.decoder(x4)
        return out
    
    def step(self, batch, batch_idx):
        x, y = batch["signal"].float(),batch["major"].float().reshape(-1,1)
        x1 = self.spectrogram_func(x).transpose(0,1).transpose(0,2)
        x1 = self.pos_encoder(x1)
        x2 = self.transformer_encoder(x1).transpose(1,0).transpose(2,1)
        x3r = self.adaptiveavgpool(x2)
        x3c = self.adaptivemaxpool(x2)
        x4 = torch.cat((x3r, x3c), dim=1)
        x4 = x4.view(x4.size(0), -1)
        out =  self.decoder(x4)
#         loss = F.binary_cross_entropy_with_logits(out, y,pos_weight=self.w_pos.to(self.device))
        loss = F.binary_cross_entropy_with_logits(out, y,)
        accuracy = torchmetrics.functional.accuracy(out,y.int(),num_classes=1)
        print(accuracy,end="\r")
        return loss, {"loss": loss,"accuracy":accuracy}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-8)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                                        optimizer,
                                        pct_start = self.pct_start,
                                        max_lr=self.max_lr,
                                        steps_per_epoch=int(len(self.train_dataloader())),
                                        epochs=self.epochs,
                                        anneal_strategy="cos",
                                        final_div_factor = 1000,
                                        max_momentum=self.max_momentum,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer],[lr_scheduler]


class Transformer_DVD(LightningModule):
    def __init__(self,attention_dropout=0.3,d_model=120,heads=30,encoding_layers=12,pool_dim=1,pct_start=0.05,max_lr=1e-4,max_momentum=0.95,epochs=50):
        super().__init__()
        dropout=0.2
        self.attention_dropout=attention_dropout
        self.d_model = d_model
        self.heads = heads
        self.encoding_layers = encoding_layers
        self.pool_dim = pool_dim
        self.pct_start = pct_start
        self.max_lr = max_lr
        self.max_momentum = max_momentum
        self.epochs = epochs
        self.spectrogram_func = torchaudio.transforms.Spectrogram(n_fft = int(self.d_model*2)-1, hop_length = 200, power = 0.2, normalized = True)
        
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.heads,
                                                        dropout = self.attention_dropout,)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoding_layers)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(self.pool_dim)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(self.pool_dim)
        
        self.decoder = nn.Sequential(
          nn.Linear(self.d_model*2*self.pool_dim,64),
          nn.Linear(64,32),
          nn.Linear(32,16),
          nn.Linear(16,4),
        )
#         d["signal"]
    def forward(self, x):
        x = x.float()
        x1 = self.spectrogram_func(x).transpose(0,1).transpose(0,2)
        x1 = self.pos_encoder(x1)
        x2 = self.transformer_encoder(x1).transpose(1,0).transpose(2,1)
        x3r = self.adaptiveavgpool(x2)
        x3c = self.adaptivemaxpool(x2)
        x4 = torch.cat((x3r, x3c), dim=1)
        x4 = x4.view(x4.size(0), -1)
        out =  self.decoder(x4)
        return out
    
    def step(self, batch, batch_idx):
        x, y = batch["signal"].float(),batch["one_hot"]
        x1 = self.spectrogram_func(x).transpose(0,1).transpose(0,2)
        x1 = self.pos_encoder(x1)
        x2 = self.transformer_encoder(x1).transpose(1,0).transpose(2,1)
        x3r = self.adaptiveavgpool(x2)
        x3c = self.adaptivemaxpool(x2)
        x4 = torch.cat((x3r, x3c), dim=1)
        x4 = x4.view(x4.size(0), -1)
        out =  self.decoder(x4)
#         loss = F.binary_cross_entropy_with_logits(out, y,pos_weight=self.w_pos.to(self.device))
        loss = F.binary_cross_entropy_with_logits(out, y,)
        auc = torchmetrics.functional.auroc(out,y.int(),num_classes=4,average="micro")
        accuracy = torchmetrics.functional.accuracy(out,y.int(),num_classes=4,average="macro")
        print(auc,accuracy,end="\r")
        return loss, {"loss": loss,"auc":auc,"accuracy":accuracy}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-8)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                                        optimizer,
                                        pct_start = self.pct_start,
                                        max_lr=self.max_lr,
                                        steps_per_epoch=int(len(self.train_dataloader())),
                                        epochs=self.epochs,
                                        anneal_strategy="cos",
                                        final_div_factor = 1000,
                                        max_momentum=self.max_momentum,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer],[lr_scheduler]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)