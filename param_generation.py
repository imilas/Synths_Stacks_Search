import json
import sounddevice as sd
import numpy as np
from numpy import random as rd
import pandas as pd 
from pippi.oscs import Osc
from pippi import dsp, noise
from pippi.soundbuffer import SoundBuffer
from helpers import *
from random import gauss
import helpers

C0=440*2**(-1*9/12)*2**(-1*4) #assume A4 is 440hz and based on it calculate our lowest note C0
sr=44100

osc_types=["sine","square","saw"]
a_d_s_r=np.arange(0,4)

#define notes based on A4=440, also used for bandpass cutoffs
num_notes=120
all_pitches=np.array([C0*2**(x/12) for x in range(0,num_notes)]) #!rename to synth pitches or something
p0_pitches=np.arange(0,num_notes,2)
p1_pitches=np.arange(0,num_notes,4)
p2_pitches=np.arange(0,num_notes,8)
p3_pitches=np.arange(0,num_notes,16)

lp0=len(p0_pitches)
lp1=len(p1_pitches)
lp2=len(p2_pitches)
lp3=len(p3_pitches)

# num_cuts=120
# all_filter_pitches=np.array([C0*2**(x/12) for x in np.arange(0,num_cuts)]) #skipping the first 2 octaves
bp_pitches=np.arange(0,num_notes,2)

num_osc_pitches=np.arange(0,4) #hard set to 4 for now
#envelope
amplitudes=np.array([0.3,0.5,0.8,1])
filter_orders=np.array([2,8,16])

squeeze_factor=10 #biasing starts towards 0
max_start=0.95 #latest start time of a sound, slightly below 1
min_length=(1-max_start)**((1/squeeze_factor))  #based on max_start, what should be min_length so that it adds up to 1?
start_spacing=10

starts=np.linspace(0,max_start,start_spacing)**squeeze_factor 
lengths=np.linspace(0,0.95,start_spacing)**squeeze_factor

class RandomParams():
    def __init__(self,name="pset"):
        self.oscType=rd.choice([0,1,2],p=[0.8,0.1,0.1])
        self.isNoise=rd.choice([0,1],p=[0.5,0.5])     
        self.A=rd.randint(len(a_d_s_r))
        self.D=rd.randint(len(a_d_s_r))
        self.S=rd.randint(len(a_d_s_r))
        self.R=rd.randint(len(a_d_s_r)) 
        self.pitch_0=rd.choice(p0_pitches)
        self.pitch_1=rd.choice(p1_pitches)
        self.pitch_2=rd.choice(p2_pitches)
        self.pitch_3=rd.choice(p3_pitches) 
        self.bpCutLow=rd.choice(bp_pitches)
        self.bpCutHigh=rd.randint(self.bpCutLow,num_notes)
        self.bpOrder=rd.randint(len(filter_orders))
        self.amplitude=rd.randint(len(amplitudes))
        self.start=rd.randint(start_spacing)
        self.length=rd.randint(start_spacing-self.start)
    def getOscType(self):
        return osc_types[self.oscType]
    def getPitches(self):
        return all_pitches[self.pitch_0],all_pitches[self.pitch_1],all_pitches[self.pitch_2],all_pitches[self.pitch_3]
    def getBandPass(self):
        return all_pitches[self.bpCutLow],all_pitches[self.bpCutHigh],filter_orders[self.bpOrder]
    def getAmp(self):
        return amplitudes[self.amplitude]
    def getLength(self):
        return starts[self.length]+min_length
    def getStart(self):
        return lengths[self.start]
    #mutate envelope with chance e and texture with chance t
    def mutate(self,e=0.75,t=0.75,s=0.8):
        if rd.rand()<e:
            self.A=rd.randint(len(a_d_s_r))
            self.D=rd.randint(len(a_d_s_r))
            self.S=rd.randint(len(a_d_s_r))
            self.R=rd.randint(len(a_d_s_r))
            self.amplitude=rd.randint(len(amplitudes))
            self.start=rd.randint(start_spacing)
            self.length=rd.randint(start_spacing-self.start)
        if rd.rand()<t:
            x0=(20*int(gauss(0,s))+self.pitch_0)%lp0
            x1=(10*int(gauss(0,s))+self.pitch_0)%lp1
            x2=(5*int(gauss(0,s))+self.pitch_0)%lp2
            x3=(2*int(gauss(0,s))+self.pitch_0)%lp3
            self.oscType=rd.choice([0,1,2],p=[0.8,0.1,0.1])
            self.pitch_0=p0_pitches[x0]
            self.pitch_1=p1_pitches[x1]
            self.pitch_2=p2_pitches[x2]
            self.pitch_3=p3_pitches[x3]
        return [e,t]
        
class Synth():
    def __init__(self,params):
        buff=SoundBuffer(channels=1)
        length=1
        if params.isNoise==1:
            buff = noise.bln(params.getOscType(),params.getLength(),30,
                150000,channels=1) 
        else:
            buff = Osc(str(params.getOscType()), freq=list(params.getPitches()),
                       channels=1).play(params.getLength()) 

        buff=buff.adsr(a=params.A, d=params.D, s=params.S, r=params.R)
        bpfilter=params.getBandPass()
        buff.frames = helpers.butter_bandpass_filter(buff.frames,bpfilter[0],bpfilter[1], 
                                                     sr, order=bpfilter[2])
        self.buff=buff
        
def ensemble(params):
    out = dsp.buffer(length=1,channels=1)
    for p in params:
        s=pg.Synth(p)
        out.dub(s.buff,p.getStart())
    return out


# #takes a  row of our scored database and returns a parameter set
# def rToParams(r,pset,n=0):
#     pset.oscType=r["oscType_%d"%(n,)]
#     pset.isNoise=r["isNoise_%d"%(n,)]
#     pset.A=r["A_%d"%(n,)]
#     pset.D=r["D_%d"%(n,)]
#     pset.S=r["S_%d"%(n,)]
#     pset.R=r["R_%d"%(n,)]
#     #pitches
#     pset.amplitude=r["amplitude_%d"%(n,)]
#     pset.bpCutLow,pset.bpCutHigh=r["bpCutLow_%d"%(n,)],r["bpCutHigh_%d"%(n,)]
#     pset.bpOrder=r["bpOrder_%d"%(n,)]
#     pset.length=r["length_%d"%(n,)]
#     pset.start=r["start_%d"%(n,)]
#     pset.pitch_0=r["pitch_0_%d"%(n,)]
#     pset.pitch_1=r["pitch_1_%d"%(n,)]
#     pset.pitch_2=r["pitch_2_%d"%(n,)]
#     pset.pitch_3=r["pitch_3_%d"%(n,)]
    
#     return pset