import random
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pippi import dsp, fx
import scipy
import param_generation as pg

sr=44100

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
        #break up the pitch list
        for i,v in enumerate(dict["pitches"]):
            dict["pitch%d"%i]=v
        del dict["pitches"]
        ##conversion to df
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
        out.dub(s.buff,p.start)
        params.append(p)
    out=fx.norm(out,1)
    return out,params

#convert pippi outputs to mono audio
def memToAud(out):
    return np.squeeze(np.asarray(out.frames))