import random
import json
import sounddevice as sd
import numpy as np
import pandas as pd 
from pippi.oscs import Osc
from pippi import dsp, noise
from pippi.soundbuffer import SoundBuffer
from helpers import *
import helpers

C0=440*2**(-1*9/12)*2**(-1*4) 
sr=44100
cutSpacing=20
lengthSpacing=10
startSpacing=10

#list of random params we choosing from
oscTypes=["sine","square","saw"]
a_d_s_r=np.arange(0,4)

#pitch stuff
numOscPitches=4 #hard set to 4 atm

#define notes based on A4=440
possiblePitches=[C0*2**(x/12) for x in range(0,120)]

amplitude=[0.3,0.5,0.8,1]
filterOrders=[2,8,16]
lengths=np.linspace(0.2,0.8,startSpacing)**0.5

class RandomParams():
    def __init__(self,name="pset"):
        self.oscType=np.random.choice([0,1,2],p=[0.5,0.25,0.25])
        self.isNoise=np.random.choice([1,0],p=[0.1,0.9])
        self.A=np.random.choice(a_d_s_r)
        self.D=np.random.choice(a_d_s_r)
        self.S=np.random.choice(a_d_s_r)
        self.R=np.random.choice(a_d_s_r)
        self.pitch_0=np.random.choice(possiblePitches)
        self.pitch_1=np.random.choice(possiblePitches)
        self.pitch_2=np.random.choice(possiblePitches)
        self.pitch_3=np.random.choice(possiblePitches)
        self.amplitude=np.random.choice(amplitude)
        self.bpCutLow,self.bpCutHigh=self.bpCut()
        self.bpOrder=np.random.choice(filterOrders,p=[0.25,0.5,0.25])
        self.length=np.random.choice(lengths)
        self.start=np.random.choice(np.linspace(0,(1-self.length),startSpacing))
    def getOscType(self):
        return oscTypes[self.oscType]
    def getPitches(self):
        return [self.pitch_0,self.pitch_1,self.pitch_2,self.pitch_3]
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
            buff = noise.bln(params.getOscType(),params.length,30,
                150000,channels=1) 
        else:
            buff = Osc(str(params.getOscType()), freq=params.getPitches(),channels=1).play(params.length) 

        buff=buff.adsr(a=params.A, d=params.D, s=params.S, r=params.R)
        buff.frames = helpers.butter_bandpass_filter(buff.frames,params.bpCutLow,params.bpCutHigh, 
                                                     sr, order=params.bpOrder)
        self.buff=buff
        
def ensemble(params):
    out = dsp.buffer(length=1,channels=1)
    for p in params:
        s=pg.Synth(p)
        out.dub(s.buff,p.start)
    return out

#takes a  row of our scored database and returns a parameter set
def rToParams(r,pset,n=0):
    pset.oscType=r["oscType_%d"%(n,)]
    pset.isNoise=r["isNoise_%d"%(n,)]
    pset.A=r["A_%d"%(n,)]
    pset.D=r["D_%d"%(n,)]
    pset.S=r["S_%d"%(n,)]
    pset.R=r["R_%d"%(n,)]
    #pitches
    pset.amplitude=r["amplitude_%d"%(n,)]
    pset.bpCutLow,pset.bpCutHigh=r["bpCutLow_%d"%(n,)],r["bpCutHigh_%d"%(n,)]
    pset.bpOrder=r["bpOrder_%d"%(n,)]
    pset.length=r["length_%d"%(n,)]
    pset.start=r["start_%d"%(n,)]
    pset.pitch_0=r["pitch_0_%d"%(n,)]
    pset.pitch_1=r["pitch_1_%d"%(n,)]
    pset.pitch_2=r["pitch_2_%d"%(n,)]
    pset.pitch_3=r["pitch_3_%d"%(n,)]
    
    return pset