from sklearn.decomposition import FastICA
from scipy.signal import detrend
import numpy as np


class EEGpreprocessing():

    def __init__(self, data, ch_range=[3,7], detrend=True, ica=True, ica_components=10, fft=True):
        self.data = data
        self.icaObj = FastICA(n_components=ica_components, whiten=True)
        self.detrend = detrend
        self.ica = ica
        self.fft = fft
        self.channels = ch_range

    def apply(self):

        data_ = []
        for d in self.data:

            d_ = d.T[:10]

            # detrend
            if self.detrend:
                d_ = detrend(d_, axis=0)

            # normalize
            d_ /= d_.std(axis=0)

            # ica
            if self.ica:
                d_ = self.icaObj.fit_transform(d_.T)

            # fft
            if self.fft:
                ch_ = []
                for j, ch in enumerate(d_.T): # fft

                    if self.channels[0] < j < self.channels[1]:
                    # if self.channels[0] == j or self.channels[1] == j:
                        fourier = np.fft.fft(ch)
                        n = len(fourier)
                        real = np.absolute(fourier)
                        half = int(n / 2.)
                        ch_.append(real[:half])

                d_ = np.array(ch_)
            data_.append(d_)

        return np.array(data_)