import numpy as np
# classes=['clap', 'guitar', 'hat', 'kick', 'piano', 'rim', 'shake', 'snare', 'synth',]
classes=['clap', 'hat', 'kick', 'snare', 'stacks'] 
classes_ranked=[c+"_rank" for c in classes]


C0=440*2**(-1*9/12)*2**(-1*4) 
sr=48000
freqSpacing=50
cutSpacing=50
lengthSpacing=10
startSpacing=10
osc_types=["sine","square","saw"]
param_cols=['oscType',
       'isNoise', 'A', 'D', 'S', 'R', 'pitch_0', 'pitch_1',
       'pitch_2', 'pitch_3', 'amplitude', 'bpCutLow', 'bpCutHigh',
       'bpOrder', 'length', 'start']
numOscPitches=4 
possiblePitches=[C0*2**(x/12) for x in range(0,120)]
amplitude=[0.3,0.5,0.8,1]
filterOrders=[2,8,16]
lengths=np.linspace(0.2,0.8,startSpacing)**0.5
