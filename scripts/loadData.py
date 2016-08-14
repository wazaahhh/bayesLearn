import pandas as pd 
from Entro import Distance
import jsonpickle as jp
import numpy as np
import csv
simple = pd.read_pickle("simple_models")
complex = pd.read_pickle("complex_models")

#print simple[simple.keys()[0]][400:403]
simple_score = {}
complex_score = {}
with open('simpleG', 'r') as f:
    simpleG = jp.decode(f.read())

with open('complexG', 'r') as f:
    complexG = jp.decode(f.read())

#print simpleG

for key in simple:

    simple_score[key] = [Distance(simpleG["probs"], t) for t in simple[key]]

for key in complex:
    complex_score[key] = [Distance(complexG["probs"], t) for t in complex[key]]

key = complex.keys()[6]

#print complex_score[key]
complex_distances = pd.DataFrame(complex_score)
complex_distances.to_csv("complex_distances.csv")    

simple_change_count_pm = {}

def count_changes(vector, error):
    count=0
    for i in range(1, len(vector)):
        if vector[i]-vector[i-1] > error:
            count+=1
    return count

def mk_change_count(list, interval):
    offset = int(float(interval)/2.0)
    counts = []
    for i in range(offset, len(list) - offset):
        counts.append(count_changes(list[i-offset:i+offset], error=0.01))
    return counts

avrg_count=[]
for key in simple:
    simple_change_count_pm[key] = np.array(mk_change_count(simple_score[key], 60))
    avrg_count.append(simple_change_count_pm[key])



std_count = np.std(np.array(avrg_count), 0)

avrg_count = np.mean(np.array(avrg_count), 0)

#print std_count




