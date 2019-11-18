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

sr=44100
stack_size=3
BATCH_SIZE=1
classes=['clap', 'guitar',
         'hat', 'kick', 'noise',
         'piano', 'rim', 'shake', 'snare', 'synth','tom', 'voc']
cDict={v:i for i,v in enumerate(classes)}

#setup CNN
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
s=torch.load("feature_extraction/models/model-4-18.states")
cnn = CNN_utils.CNN_net()
cnn.load_state_dict(s["model_state_dict"])
cnn.to(device)

def makeRows():

    ## function that makes a row of parameters and the scores for the parameters 
    ## this row can then be added to a dataframe/csv file etc
    out,params=hp.stackMaker(1)
    a=hp.memToAud(out)
    
    # get the image for that audio
    
    try:
        im=mu.audToImage(a,128)
    except:
        return "fail"
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
    rowDict=dict(zip(classes,o_norm))
    
    df=pd.concat([pd.DataFrame.from_dict([rowDict]),hp.paramToDF(params)],axis=1)    
    x=df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in x]
    return vals

while (1):
    print(makeRows())
