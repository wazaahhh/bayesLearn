import pandas as pd 
from Entro import Distance
import jsonpickle as jp

simple = pd.read_pickle("simple_models")
complex = pd.read_pickle("complex_models")

simple_score = {}

with open('simpleG', 'r') as f:
    simpleG = jp.decode(f.read())

with open('complexG', 'r') as f:
    complexG = jp.decode(f.read())

for key in simple:

    simple_score[key] = [Distance(simpleG.probs, t) for t in simple[key]]
