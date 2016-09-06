import scipy as sp
from math import sqrt
from numpy.random import random
from scipy.stats.stats import pearsonr

truth =random( 8 ); truth = truth/sum(truth) 

#with a normalization so that all entries add to 1.0. 


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def quadloss(act, pred):
    return sqrt(sum([(act[i]-pred[i])**2 for i in range(len(pred))]))

ll = []
ql =[]

for i in range(1000):
    model = random(8)
    model = random(8)
    ll.append(logloss(truth, model))
    ql.append(quadloss(truth, model))

print(pearsonr(ll, ql))
