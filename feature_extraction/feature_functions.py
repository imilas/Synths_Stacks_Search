import madmom 
import numpy as np
import seaborn as sns
import pandas as pd
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import librosa, librosa.display
sr=40000

def fitFreq(rows,frameLen=20,hopLen=999,numFrames=100,exponents=4):
    rows.reset_index()
    hopLen=frameLen-1
    def getFreqFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=40000,
            frame_size=frameLen,hop_size=hopLen)
        feat=np.zeros(frameLen)
        for frame in fs:
            X=np.absolute(scipy.fft(frame))
            feat+=X
        return sr*feat/len(rows)
    sample_features=[]
    for i,s in rows.iterrows():
        features=getFreqFeat(s["audio"])
        zc,sc=zc_center(s["audio"])
        l=len(s["audio"])
        others=np.array([zc,sc,l])
        shape=lineApproximation(s["audio"])
        f=np.concatenate([features,others,shape],axis=None)
        sample_features.append(f)
    
    df=pd.DataFrame(sample_features)
    feat_cols=[ 'freq'+str(i) for i in range(frameLen)]
    feat_cols.extend(["zc","sc","l"])
    feat_cols.extend( ['env_shape'+str(i) for i in range(exponents+1)])
    df.columns=feat_cols
    df["label"]=rows.reset_index().label
    df["path"]=rows.reset_index().path
    return df

def zc_center(signal):
    if len(signal)>1000:
        signal=signal[0:1000]
        l=signal.shape[0]
        zc=librosa.feature.zero_crossing_rate(signal,frame_length=l+1, hop_length=l,)[0,0],
        spec_center=librosa.feature.spectral_centroid(signal,sr=40000,n_fft=l+1, hop_length=l)[0,0],
        zcl=np.log(zc[0]+0.01)
        scl=spec_center[0]
        ret=[scl,zcl]
        return ret
    else:
        return [0,0]

def lineApproximation(x,expo=4):
    fl=int(len(x)/100)
    hl=fl
    rmse = librosa.feature.rms(x, frame_length=fl, hop_length=hl, center=True)
    librosa.display.waveplot(x, sr=sr)
    frames = range(rmse.shape[1])
    ft = librosa.frames_to_time(frames, sr=sr, hop_length=hl)
    x=ft
    y=rmse[0]
    z = np.polyfit(x, y, expo)
    f = np.poly1d(z)
    return z

def fitMels(signals,t="unknown_drum",p="path",num_feats=2):
    def getFeat(x):
        X = librosa.feature.melspectrogram(S=x,n_mels=num_feats, sr=sr,)
        return X
    feats=[]
    for s in signals:
        features=getFeat(s)
        feats.append(features)
    df=pd.DataFrame(feats)
    feat_cols=[ 'feat'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    df["path"]=p
    return df

    
def getOnsetDF(signals,t="u"):
    def getOnsets(x):
        spec = madmom.audio.spectrogram.Spectrogram(x, frame_size=300, hop_size=300)
        X=madmom.features.onsets.high_frequency_content(spec)
        return X[0:100] 
    onsets=[]
    for s in signals:
        onset=getOnsets(s)
        onsets.append(onset[0:300])
    
    df=pd.DataFrame(onsets)
    feat_cols=['onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df

def fitPolyWave(signals,t="unknown_drum",frameLen=2000,hopLen=1600,polyDeg=2,num_feats=100):
    def getFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=40000,
            frame_size=frameLen,hop_size=hopLen)
        feats=[]
        for frame in fs[0:num_feats]:
            try:
                feat=np.polyfit(frame,np.linspace(0,1,frame.shape[0]),deg=polyDeg)
                feats.extend(feat)
            except:
                print("bad frame")
                continue
        return feats[0:num_feats] 
    analyzed=[]
    for s in signals:
        features=getFeat(s)
        analyzed.append(features)
    
    df=pd.DataFrame(analyzed)
    feat_cols=[ 'onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df



