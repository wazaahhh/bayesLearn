import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import jsonpickle as jp
from Entro import Distance
simple = pd.read_pickle("simple_models")
complex = pd.read_pickle("complex_models")

simple_score = {}
complex_score = {}

with open('simpleG', 'r') as f:
    simpleG = jp.decode(f.read())

with open('complexG', 'r') as f:
    complexG = jp.decode(f.read())

for key in simple:
    simple_score[key] = [Distance(simpleG["probs"], t) for t in simple[key]]

for key in complex:
    complex_score[key] = [Distance(complexG["probs"], t) for t in complex[key]]

simple_distances = pd.DataFrame(simple_score)
complex_distances = pd.DataFrame(complex_score)

color = ["green","red"]
legend = ["simple","complex"]

pl.close()
for k,kx  in enumerate([simple_distances,complex_distances]):
    #array = np.array(dfSimpleLearning)
    ar = np.array(kx)
    mean = []
    median = []
    std = []

    for i,ix in enumerate(ar):
        cond = np.logical_not(np.isnan(ix))
        median = np.append(median,np.median(ix[cond]))
        mean  = np.append(mean,np.mean(ix[cond]))
        std = np.append(std,np.std(ix[cond]))

        #pl.plot(median)
    pl.figure(1)
    pl.plot(mean,color=color[k],label=legend[k])
    pl.plot(mean+std,'--',color=color[k])
    pl.plot(mean-std,'--',color=color[k])

#pl.figure(2)
#plotFit(mean)

pl.xlim(0,2500)
pl.legend(loc=0)
pl.savefig('foo.eps')
#pl.show()
