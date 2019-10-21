#!/usr/bin/env python
import time
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from matplotlib import pyplot as plt
import pandas
import numpy
import os
import random
n_neighbors = 10
data = pandas.read_csv("3_stack.csv",header=None)
# print(data.head())
# X = data.iloc[:,44:].as_matrix()
# X = data.iloc[:,1:88:2].as_matrix()
X = data.iloc[:,1:88].as_matrix()
X = numpy.nan_to_num(X)
# print(X)
time_start = time.time()
# tsne = TSNE(n_components=2, random_state=0, perplexity=100, verbose=1)
# tsne = Isomap(n_neighbors, n_components=2)
# tsne = LocallyLinearEmbedding(n_neighbors,n_components=2)
tsne = SpectralEmbedding(n_neighbors=n_neighbors,n_components=2,random_state=0)
print("# Fitting tsne")
X_2d = tsne.fit_transform(X)
print("# Plotting")
print('# t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fig, ax = plt.subplots()
ax.scatter(X_2d[:, 0], X_2d[:, 1], picker=5 )

def on_pick(event):
    line = event.artist
    #print(line)
    #print("[%s]" % event.ind)
    i = random.choice(event.ind)
    filename = data.iloc[i,0]
    print("play %s &" % filename)
    os.system("play %s &" % filename)
    #print(x[event.ind[0]], y[event.ind[0]])
    #xdata, ydata = line.get_data()
    #ind = event.ind
    #print('on pick line:', np.array([xdata[ind], ydata[ind]]).T)

cid = fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()

