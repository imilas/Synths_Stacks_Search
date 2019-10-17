from pippi.oscs import Osc, Osc2d, Pulsar, Pulsar2d, Alias, Bar
from pippi import dsp, interpolation, wavetables, fx, oscs,soundpipe
from pippi.soundbuffer import SoundBuffer
from pippi.wavesets import Waveset
from pippi import dsp, fx
import random
import sounddevice as sd
from IPython.display import Audio
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pippi import dsp, noise
import scipy
sr=44100
plt.figure(figsize=(8, 5))

def specShow(sig):
    # multiframe spectrogram
    #make mono
    try:
        sig=sig.frames
    except:
        pass
    print(sig)
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
    # make mono
    X = scipy.fft(sig)
    X_mag = librosa.core.amplitude_to_db(np.absolute(X))
    f = np.linspace(0, sr, len(X_mag)) # frequency variable
    plt.subplot(1, 2, 2)
    res=int(len(sig)/2)
    plt.plot(f[:res], X_mag[:res])
#     plt.plot(f, X_mag)
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
