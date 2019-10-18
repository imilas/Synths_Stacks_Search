import random
import json
import sounddevice as sd
import numpy as np
import pandas as pd 
from pippi.oscs import Osc
from pippi import dsp, noise
from pippi.soundbuffer import SoundBuffer
from helpers import *

sr=44100
freqSpacing=50
cutSpacing=50
lengthSpacing=10
startSpacing=10

#list of random params we choosing from
oscTypes=["sine","square","saw"]
a_d_s_r=np.arange(0,4)
#pitch stuff
numOscPitches=5
possibleInitPitches=np.geomspace(30,15000,freqSpacing)

amplitude=[0.3,0.5,0.8,1]
filterOrders=[2,8,16]
lengths=np.linspace(0.2,0.8,startSpacing)**0.5

class RandomParams():
    def __init__(self,name="pset"):
        #####type
        self.oscType=np.random.choice(oscTypes,p=[0.5,0.25,0.25])
        self.isNoise=np.random.choice([1,0],p=[0.1,0.9])
        self.A=np.random.choice(a_d_s_r)
        self.D=np.random.choice(a_d_s_r)
        self.S=np.random.choice(a_d_s_r)
        self.R=np.random.choice(a_d_s_r)
        #pitches
        self.numOscPitches=numOscPitches
        self.initPitch=np.random.choice(possibleInitPitches)
        self.pitchPathMag=np.random.choice([-1,0,1])
        self.pitchPathAccel=np.random.choice([0,2,8])
        self.pitches=self.pitchSelection(self.numOscPitches)
        #######
        self.amplitude=np.random.choice(amplitude)
        self.bpCutLow,self.bpCutHigh=self.bpCut()
        self.bpOrder=np.random.choice(filterOrders,p=[0.25,0.25,0.5])
        self.length=np.random.choice(lengths)
        self.start=np.random.choice(np.linspace(0,(1-self.length),startSpacing))

    def pitchSelection(self,n=1):
        pList=(self.initPitch)+(15000*self.pitchPathMag*(np.linspace(0,1,numOscPitches)**self.pitchPathAccel))
        z=np.clip([int(x) for x in pList],30,15000)
        return list(z)

    def bpCut(self):
        filterCutoffs=np.geomspace(30,15000,cutSpacing)
        cutIndex=np.arange(0,cutSpacing)
        highIndex=np.random.choice(cutIndex)
        lowIndex=np.random.choice(cutIndex[int(highIndex):])
        return filterCutoffs[[highIndex,lowIndex]]

class Synth():
    def __init__(self,params):
        buff=SoundBuffer(channels=1)
        length=1
        if params.isNoise==1:
            buff = noise.bln(str(params.oscType),params.length,30,
                150000,channels=1) 
        else:
            buff = Osc(str(params.oscType), 
                freq=params.pitches,channels=1).play(params.length) 

        buff=buff.adsr(a=params.A, d=params.D, s=params.S, r=params.R)
        buff.frames = butter_bandpass_filter(buff.frames,params.bpCutLow,params.bpCutHigh, 
                                                sr, order=params.bpOrder)
        self.buff=buff

# out = dsp.buffer(length=1)
# for i in range(1): 
#     p=RandomParams()
#     print(p.__dict__)
#     # s=Synth(p)
#     out.dub(s.buff,p.start)
#     out=fx.norm(out,1)

    
# sd.play(out)
# specShow(out)
# print(p.pitches)
