import pickle
import numpy as np
import pandas as pd

import os, sys, random, math, time, socket, struct

import scipy.signal

bufferpath = "../../python/signalProc"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),bufferpath))
import bufhelp

with open('Merian.pickle', 'rb') as file:
    data = pickle.load(file)

import matplotlib.pyplot as plt
m = np.asmatrix(data['data'][20])
m = m[:, 0:9]

m = scipy.signal.detrend(m, axis=1)

plt.plot(m)
plt.savefig('test.png')
