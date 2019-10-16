import random
import json
import sounddevice as sd
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from pippi import dsp, noise
import scipy

sr=44100
numFR=50
numCut=50
oscTypes=["sine","square","saw","noise"]
a_d_s_r=np.arange(0,4)
numOscPitches=np.arange(1,4)
pitchFreq=np.geomspace(30,15000,numFR)
amplitude=np.linspace(0,1,10)

class SynthParams():
    def __init__(self,name="pset"):
        self.oscTypes=np.random.choice(oscTypes,p=[0.3,0.2,0.1,0.4])
        self.A=np.random.choice(a_d_s_r)
        self.D=np.random.choice(a_d_s_r)
        self.S=np.random.choice(a_d_s_r)
        self.R=np.random.choice(a_d_s_r)
        self.numOscPitches=np.random.choice(numOscPitches)
        self.pitchFreq=np.random.choice(pitchFreq)
        self.amplitude=np.random.choice(amplitude)
        self.bpCuts=bpCut()

    def bpCut():
        filterCutoffs=np.geomspace(30,15000,numCut)
        cutIndex=np.arange(0,numCut)
        highIndex=np.random.choice(cutIndex)
        lowIndex=np.random.choice(cutIndex[highIndex:])
        return filterCutoffs[[highIndex,lowIndex]]
