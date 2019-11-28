import param_generation as pg
import helpers as hp
import imp
imp.reload(pg)
imp.reload(hp)
###
import numpy as np
import pandas as pd
import torch
import torchvision
from feature_extraction import CNN_utils
####
from pippi import dsp, noise
from feature_extraction import mir_utils as mu
import librosa
import torch.utils.data as utils
import torchvision.transforms as transforms
from PIL import Image
import scipy.stats as ss
classes=['clap', 'guitar',
         'hat', 'kick', 'noise',
         'piano', 'rim', 'shake', 'snare', 'synth','tom', 'voc']
classes_ranked=[c+"_rank" for c in classes]
cDict={v:i for i,v in enumerate(classes)}

sr=41000
score_cols=['clap', 'guitar', 'hat', 'kick', 'noise', 'piano', 'rim', 'shake', 'snare', 'synth', 'tom', 'voc']
classes_ranked=[c+"_rank" for c in score_cols]

device="cpu"
s=torch.load("feature_extraction/models/model-4-18.states")
cnn = CNN_utils.CNN_net()
cnn.to(device)
cnn.load_state_dict(s["model_state_dict"])

#takes a series of params a series of scores
def rank_score(r):
    params=hp.rToParams(r)
    out=pg.Synth(params)
    out = dsp.buffer(length=1,channels=1)
    s=pg.Synth(params)
    out.dub(s.buff,params.start)
    a=hp.memToAud(out)
    try:
        im=mu.audToImage(a,128)
    except:
        return -10
    z=librosa.util.normalize(im)
    t= transforms.Compose(
        [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
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
    df=pd.concat([pd.DataFrame.from_dict([rank_dict]),pd.DataFrame.from_dict([score_dict]),hp.paramToDF([params])],axis=1)    
    
    return df.iloc[0]