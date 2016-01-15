from numpy.random import random
from DataHandler import simpleG, complexG, graph_tools
from Entro import Diversity
import numpy as np

graph_tools(simpleG)
graph_tools(complexG)

simpleTruth = simpleG.probs
n_s =len(simpleTruth)

complexTruth = complexG.probs
n_c = len(complexTruth)

def simple_random():
    random_vec =random(n_s)
    normalizer=sum(random_vec)
    return [(i+1)/(normalizer+n_s) for i in random_vec]

def complex_random():
    random_vec =random(n_c)
    normalizer=sum(random_vec)
    return [(i+1)/(normalizer+n_c) for i in random_vec]


def exploration(series):
    
    explor=[0]
    for t in range(len(series)):
        if not hasattr(series[t], "__len__"):
            explor.append(np.nan)
        else:
            models=[]
            for tau in range(t):
                if not hasattr(series[tau], "__len__"):
                    continue
                else:
                    models.append(np.array(series[tau]))
            
            if models:
                explor.append(Diversity([list(np.sum(np.array(models), 0)/float(len(models))), list(series[t])]))
            else:
                explor.append(np.nan)

    return explor

simple_random_series=[simple_random() for i in range(1000)]

complex_random_series = [complex_random() for i in range(1000)]

simple_random_explore = exploration(simple_random_series)

simple_random_score = [Diversity([model, simpleTruth]) for model in simple_random_series] 

def simple_draw_sample(n):
    def draw_one():
        draw=simpleG.draw_samples()[0]
        
        return [draw[name] for name in simpleG.names]
    return [tuple(draw_one()) for i in range(n)]


def simple_rational(observations):
    joint=np.ones(len(simpleG.cross_prod))
    for item in observations:
        index = simpleG.cross_prod.index(item)
        joint[index]+=1
    norm = sum(joint)
    return joint/float(norm)
