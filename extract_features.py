from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import os, tempfile, warnings
import numpy as np
import argparse
import warnings
from multiprocessing import Pool

def add_feature(mfcc1, rmsa1):
    tmfcc1 = np.zeros((mfcc1.shape[0],mfcc1.shape[1]+rmsa1.shape[0]))
    n = mfcc1.shape[0]
    m = mfcc1.shape[1]
    w = rmsa1.shape[0]
    tmfcc1[0:n,0:m] = mfcc1[0:n,0:m]
    tmfcc1[0:n,m:m+w]   = np.transpose(rmsa1[0:w,0:n])
    return tmfcc1

def mfcc(audio, nwin=256, nfft=512, fs=16000, nceps=13):
    #return librosa.feature.mfcc(y=audio, sr=44100, hop_length=nwin, n_mfcc=nceps)
    return [np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_fft=nfft, win_length=nwin,n_mfcc=nceps))]

def std_mfcc(mfcc):
    return (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

def extract_features_from_file(filename):
    audio = wavfile.read(filename, mmap=True)[1] / (2.0 ** 15)
    audio=audio[0:44100]
    return extract_features(audio,fs=44100)

def extract_features(buffer,fs=44100):
    sr = fs
    a1 = buffer
    # print(a1.shape)
    mfcc1 = mfcc(a1, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    mfcc1 = std_mfcc(mfcc1)
    rmsa1 = librosa.feature.rms(a1)
    cent1 = librosa.feature.spectral_centroid(y=a1, sr=fs)
    rolloff1 = librosa.feature.spectral_rolloff(y=a1, sr=fs, roll_percent=0.1)
    chroma_cq1 = librosa.feature.chroma_cqt(y=a1, sr=fs, n_chroma=10)
    onset_env1 = librosa.onset.onset_strength(y=a1, sr=sr)
    pulse1 = librosa.beat.plp(onset_envelope=onset_env1, sr=sr)
    mfcc1 = add_feature(mfcc1, rmsa1)
    mfcc1 = add_feature(mfcc1, rolloff1/fs)
    mfcc1 = add_feature(mfcc1, cent1/fs)
    mfcc1 = add_feature(mfcc1, chroma_cq1)
    mfcc1 = add_feature(mfcc1, onset_env1.reshape(1,onset_env1.shape[0]))
    mfcc1 = add_feature(mfcc1, pulse1.reshape(1,onset_env1.shape[0]))
    return mfcc1

def uniform_features(features,n=40):
    s = features.shape
    o = np.zeros(s[0] * n)
    N = s[0]*n
    o[0:N] = features.reshape((s[0] * s[1],))[0:N]
    return o

def mean_features(features):
    return np.mean(features, axis=1)

def test_driver():    
    fname = "amir/amir/3_stack/sounds/1008.wav"
    features = extract_features_from_file(fname)
    print(features.shape)
    print(features)
    plt.imshow(features)
    plt.show()
    ufeatures = uniform_features(features)
    print(ufeatures.shape)
    print(ufeatures)
    mfeatures = mean_features(features)
    print(mfeatures.shape)
    print(mfeatures)
    plt.imshow(mfeatures.reshape((29,3)))
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Process some sounds.')
    parser.add_argument('sounds', metavar='N', type=str, nargs='+',
                    help='path to Sounds to load')
    args = parser.parse_args()
    return args
def makeCSV(sound):
    try:
        row = mean_features(extract_features_from_file(sound))
        print(",".join(['"'+sound+'"', ",".join([str(x) for x in row])]))
    except Exception as e:
        warnings.warn(str(e))
    
if __name__ == "__main__":
    args = parse_args()
    sounds = args.sounds
    sounds.sort()
    pool = Pool()                         # Create a multiprocessing Pool
    pool.map(makeCSV,sounds)  
    # test_driver()
