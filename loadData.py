import pandas as pd 

simple = pd.read_pickle("simple_models")
complex = pd.read_pickle("complex_models")

print simple.keys()
