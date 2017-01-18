import pickle
import numpy as np
import pandas as pd

import os, sys, random, math, time, socket, struct
bufferpath = "../../python/signalProc"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),bufferpath))
import bufhelp

with open('Merian.pickle', 'rb') as file:
    data = pickle.load(file)

print(data['events'][0])
df = pd.DataFrame(np.asmatrix(data['data'][0]))
for i in range(37):
    print(df[i].describe())