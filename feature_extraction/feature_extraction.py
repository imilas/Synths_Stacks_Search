import numpy as np
import pandas as pd
import seaborn as sns
import scipy, matplotlib.pyplot as plt
from pathlib import Path
import mir_utils as miru
import feature_functions as ff
import scipy
import librosa,librosa.display
import madmom
import pandas as pd
import multiprocessing

df=miru.audioFrames(load=True)
df=df.sample(1000)
def getFeats(d):
    d_copy=d.copy()
    ffd=ff.fitFreq(d_copy)
    return ffd
    
num_processes = 2
chunks = np.array_split(df,num_processes)
pool = multiprocessing.Pool(processes=num_processes)
result = pool.map(getFeats, chunks)
feats=pd.concat(result)

print(feats)

#ddrndperm = np.random.permutation(feats.shape[0])
feats.to_csv("../csvs/feat_frequency_bins.csv",index=False)
