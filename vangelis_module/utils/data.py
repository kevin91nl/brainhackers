import utils
from utils.common import project_root,window,training_dataset
import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import fft
import pickle


class dataset():

    def __init__(self):
        N=190
        self._All_Frames=[]
        self._TOTAL_X=np.zeros((1620,int(N/window()),9*window()))
        self._TOTAL_Y=np.zeros((1620,int(N/window()),4))
        with np.load(training_dataset()) as data:
            ravel= np.arange(1620)
            np.random.shuffle(ravel)
            X=data['arr_0'][ravel]
            y=data['arr_1'][ravel]
            x_temp=np.zeros((9,N))
            for sample in range(X.shape[0]):
                frame_temp=np.zeros((9,N/window(),window()))
                for electrode in range(x_temp.shape[0]):
                    x_temp[electrode,:188]=X[sample,electrode*188:(electrode+1)*188]
                    frame_temp[electrode]= x_temp[electrode].reshape(int(N/window()),window())
                for frame in range(frame_temp.shape[1]):
                    FRAME_Normal=np.zeros((90,))
                    for electrode in range(frame_temp.shape[0]):
                        FRAME=np.absolute(fft(frame_temp[electrode,frame], axis=0))
                        FRAME_Normal[electrode*10:(electrode+1)*10] = FRAME/np.linalg.norm(FRAME)
                    self._All_Frames.append(FRAME_Normal)
                    self._TOTAL_X[sample,frame,:]=FRAME_Normal
                    self._TOTAL_Y[sample,frame,:]=y[sample]
        self._All_Frames=np.array(self._All_Frames)



    @property
    def X(self):
        return self._TOTAL_X

    @property
    def Y(self):
        return self._TOTAL_Y

    @property
    def Frames(self):
        return self._All_Frames