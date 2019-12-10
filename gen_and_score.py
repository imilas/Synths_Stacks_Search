import numpy as np
import pandas as pd
import torch
import torchvision
from feature_extraction import CNN_utils
import matplotlib.pyplot as plt
import librosa
# generation imports
from pippi.soundbuffer import SoundBuffer
from pippi import dsp,fx
import helpers as hp

import param_generation as pg
import _pickle as pickle
from IPython.display import Audio
from feature_extraction import mir_utils as mu
###
import torch.utils.data as utils
import torchvision.transforms as transforms
from PIL import Image
###
import scipy.stats as ss
import common_vars as comv
import imp
import uuid

dump_file=str(uuid.uuid4())[0:4] #lazy way of generating unique file name

imp.reload(comv)

sr=44100
stack_size=3
BATCH_SIZE=1
classes=comv.classes
classes_ranked=comv.classes_ranked
cDict={v:i for i,v in enumerate(classes)}

#setup CNN
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
s=torch.load("./feature_extraction/models/model-4-19.states")
cnn = CNN_utils.CNN_net()
cnn.to(device)
cnn.load_state_dict(s["model_state_dict"])

def rank_score():
    ## function that makes a row of parameters and the scores for the parameters 
    ## this row can then be added to a dataframe/csv file etc
    out,params=hp.stackMaker(1)
    a=hp.memToAud(out)
    
    # get the image for that audio
    try:
        im=mu.audToImage(a,128)
    except:
        return rank_score()
    z=librosa.util.normalize(im)
    t= transforms.Compose(
        [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    #normalize array->pilform ->apply transoforms,
    z=(((z - z.min()) / (z.max() - z.min())) * 255.9).astype(np.uint8)
    zi=Image.fromarray(z)
    z=t(zi)
    images=z.reshape([1,1,128,128])

    dimg=images.to(device)
    outputs=cnn(dimg)
    _, predicted = torch.max(outputs, 1)

    o=outputs.cpu().detach().numpy()[0]
    o_norm=o-min(o)
    o_norm=o_norm/sum(o_norm)
    score_dict=dict(zip(classes,o_norm))
    #ranks based on score
    ranks=1+len(classes_ranked)-ss.rankdata(o_norm) 
    rank_dict=dict(zip(classes_ranked,ranks))
    df=pd.concat([pd.DataFrame.from_dict([rank_dict]),pd.DataFrame.from_dict([score_dict]),hp.paramToDF(params)],axis=1)    
    
    return df

#write once with the header, no headers afterwards
df=rank_score()
df.to_csv("csvs/%s.csv"%(dump_file,), index=None, sep=',', mode='a')



num_iter=5000
dump_iter=int(num_iter/20)+1 #dump csv every dump_iter iteration
for i in range(1,num_iter+1):
    df=pd.concat([df,rank_score()])  
    if i%dump_iter==0:
        df.to_csv("csvs/%s.csv"%(dump_file,),header=None, index=None, sep=',', mode='a')
        df=rank_score()


