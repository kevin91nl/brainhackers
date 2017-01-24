import utils
from utils.common import project_root,window,training_dataset


import numpy as np
from scipy.fftpack import fft
import pickle


class dataset():

    def __init__(self):
        N=200
        temp_x=np.zeros((N,))
        self._All_Frames=[]
        self._TOTAL_X=np.zeros((1620,int(N/window()),window()))
        self._TOTAL_Y=np.zeros((1620,int(N/window()),4))
        with open(training_dataset(), 'rb') as file:
            self._X, self._y = pickle.load(file)
            XX=np.asarray(self._X)
            yy=np.asarray(self._y)
            for index in range(XX.shape[0]):
                temp_x[:198]=XX[index,:]
                temp_x_=temp_x.reshape(int(N/window()),window())
                for frame in range(temp_x_.shape[0]):
                    FRAME=np.absolute(fft(temp_x_[frame], axis=0))
                    self._All_Frames.append(FRAME)
                    self._TOTAL_X[index,frame,:]=FRAME
                    self._TOTAL_Y[index,frame,:]=yy[index]
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