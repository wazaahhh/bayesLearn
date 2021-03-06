import time
import numpy as np
from math import log, sqrt, exp
from numpy import array as arr

def is_nan(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return True
    else: return False

def MakeMixture(one, two, lamb=0.5):
    """
    if the order of names is different between to different distributions, the probability vectors must be first reordered
    Pr1, Pr2 are two probability distributions, implemented as simple ordered lists
    """
    
    if (len(one)==len(two)):
        return [(lamb*a+(1-lamb)*b) for a, b in zip(one, two)]
    else: 
        raise IndexError

def KL_divergence(first, second):
    """ 
    Compute KL divergence of two probability vectors, expressed as python lists.
    The arguments are called first and second because the 
    KL_divergence is assymetric, so that KL_divergence(first, second)!=KL_divergence(second, first)
    """
    
    if (len(first)==len(second)):
        probs1=[float(item)/sum(first) for item in first]
        probs2=[float(item)/sum(second) for item in second]
    else: 
        raise IndexError
    try:
        return sum(p * log((p /q)) for p, q in zip(probs1, probs2) if p != 0.0 or p != 0)
    except ZeroDivisionError:
        return float("inf")


def JensenShannonDivergence(g, h):
    '''
    g and h can again be dictionaries or lists, but order does not matter here
    JensenShannonDivergence(g, h)==JensenShannonDivergence(h, g)
    '''
    JSD = 0.0
    lamb=0.5
    mix=MakeMixture(g, h, lamb)
    JSD=lamb * KL_divergence(g, mix) + (1-lamb) * KL_divergence(h, mix)
    return JSD

def Distance(a, b):
    if is_nan(a) or is_nan(b): return np.nan

    return sqrt(JensenShannonDivergence(a, b))

def Entropy(pr):
    try:
        vals=pr.values()
    except:
        vals=pr
    tot=1.0/sum(vals)
    return -sum([float(p)*tot*log(float(p)*tot, 2) for p in vals if p != 0.0 or p != 0])


def MixedN(ls):
    """
    ls: a list of either lists or dictionaries.
    """
    
    if (len(ls)==1):
        if type(ls[0])==list:
            return [item/float(sum(ls[0])) for item in ls[0]]
        elif type(ls[0])==dict:
            return {key:value/float(sum(ls[0].values())) for key, value in ls[0].items()}

    lamb = 1.0/len(ls)
    if (sum([type(it)==list for it in ls])==len(ls)):
        total=arr([0]*len(ls[0]));
        for it in ls:
            total= total + arr([n/float(sum(it)) for n in it])
        mix = total*lamb
        return mix

    elif (sum([type(it)==dict for it in ls])==len(ls)):
        keys=set([])
        for it in ls:
            keys.update(set(it.keys()))
        mix={key:sum([(float(1)/sum(it.values()))*it.get(key, 0)*lamb for it in ls]) for key in keys}
        return mix
            

def N_point_JSD(ls):
    mix=MixedN(ls)
    
    try:
        keys=set(mix.keys())
        orderedList=[[g.get(key, 0)/float(sum(g.values())) for key in keys] for g in ls]
    except:
        orderedList=ls

    return Entropy(mix) - (1.0/len(ls))*sum([Entropy(g) for g in orderedList])
    
def Diversity(ls):
    ls=[it for it in ls if type(it)!=float]
    n=len(ls)
    try:
        return sqrt(N_point_JSD(ls)/log(n, 2))
    except: 
        return np.nan
