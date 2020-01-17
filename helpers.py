import random
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pippi import dsp, fx
import scipy
import param_generation as pg
        
from common_vars import sr
import string
def specShow(sig):
    plt.figure(figsize=(8, 5))
    # multiframe spectrogram
    #make mono
    try:
        sig=sig.frames
    except:
        pass
    sig=np.nan_to_num(list(sig))
    try:
        sig=lib.to_mono(np.transpose(sig))
    except:
        return
    X = lib.stft(sig)
    Xdb = lib.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    lib.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

    # single frame spectrogram
    X = scipy.fft(sig)
    X_mag = librosa.core.amplitude_to_db(np.absolute(X))
    f = np.linspace(0, sr, len(X_mag)) # frequency variable
    plt.subplot(1, 2, 2)
    res=int(len(sig)/2)
    plt.plot(f[:res], X_mag[:res])
    plt.xlabel('Frequency (Hz)')
    
def waveShow(sig):
    try:
        sig=sig.frames
    except:
        pass
    sig=lib.to_mono(np.transpose(sig)) 
    lib.display.waveplot(sig)

    
from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(sig, cutoff, fs, order=5):
    try:
        sig=sig.frames
    except:
        pass
    sig=lib.to_mono(np.transpose(sig)) 
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, sig)
    return np.reshape(y,(-1,1))

from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
        try:
            sig=sig.frames
        except:
            pass
        sig=lib.to_mono(np.transpose(sig)) 
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, sig)
        return np.reshape(y,(-1,1))

def paramToDF(params):
    pdfs=[]
    for j,p in enumerate(params):
        dict=p.__dict__.copy() 
        pdfs.append(pd.DataFrame.from_dict([dict]).add_suffix("_%d"%j))    
    df=pd.concat(pdfs,axis=1)
    return df

def paramToSound(params):
    out = dsp.buffer(length=1,channels=1)
    for p in params:
        s=pg.Synth(p)
        out.dub(s.buff,p.start)
    return fx.norm(out,1)

def stackMaker(n,l=1,c=1):
    #makes a sample of length l with num channels c
    out = dsp.buffer(length=l,channels=c)
    params=[]
    for i in range(n): 
        p=pg.RandomParams()
        s=pg.Synth(p)
        out.dub(s.buff,p.getStart())
        params.append(p)
    out=fx.norm(out,1)
    return out,params


def rToParams(r,n=0):
    r=r.astype(int)
    pset=pg.RandomParams()
    pset.oscType=int(round(r["oscType_%d"%(n,)]))
    pset.isNoise=int(round(r["isNoise_%d"%(n,)]))
    pset.A=r["A_%d"%(n,)]
    pset.D=r["D_%d"%(n,)]
    pset.S=r["S_%d"%(n,)]
    pset.R=r["R_%d"%(n,)]
    #pitches
    pset.pitch_0=int(r["pitch_0_%d"%(n,)])
    pset.pitch_1=int(r["pitch_1_%d"%(n,)])
    pset.pitch_2=int(r["pitch_2_%d"%(n,)])
    pset.pitch_3=int(r["pitch_3_%d"%(n,)])
    #######
    pset.amplitude=r["amplitude_%d"%(n,)]
    pset.bpCutLow,pset.bpCutHigh=r["bpCutLow_%d"%(n,)],r["bpCutHigh_%d"%(n,)]
    pset.bpOrder=int(round(r["bpOrder_%d"%(n,)]))
    pset.start=r["start_%d"%(n,)]
    pset.length=r["length_%d"%(n,)]
    return pset

#convert pippi outputs to mono audio
def memToAud(out):
    return np.squeeze(np.asarray(out.frames))

# converting audio to images
def cutAudio(x,num_samples=sr):
    xt,i=lib.effects.trim(x, top_db=50)
    nTrimmed=len(x)-len(xt)
    xt=xt[0:num_samples]
    xt=lib.util.normalize(xt)
    new_x =np.pad(xt,(0,num_samples-xt.shape[0]),'constant')
    return new_x,nTrimmed

def audToImage(x,num_bins=100):
    D=librosa.stft(x,n_fft=num_bins**2,win_length=num_bins**2,hop_length=int(num_bins*4)+1)
    S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=num_bins)
    S_dB = librosa.power_to_db(np.abs(S))
    return S_dB

#random string generation
def string_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))