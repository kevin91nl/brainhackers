from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

import sys
import os
import pickle
import numpy as np
from make_labels import LabelMarker

bufferpath = "../../python/signalProc"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), bufferpath))


# read data
with open('all_Xy.pickle', 'rb') as file:
    X, y = pickle.load(file)

    X_ = X[y[:, 0] == 1]
    y_ = y[y[:, 0] == 1]
    X_ = np.array(X_).reshape(len(X_), -1)

    clusters = KMeans(n_clusters=10).fit(X_)
    labels = clusters.labels_

    # X_r = X[y[:, 1] == 1]
    # y_r = y[y[:, 1] == 1]
    # X_r = np.array(X_r).reshape(len(X_r), -1)
    #
    # clusters_r = KMeans(n_clusters=9).fit(X_r)
    # labels_r = clusters.labels_

    # make them again
    lm = LabelMarker(X, y, labels)
    hot = lm.kmeans()
    print X.shape, y.shape
    print hot.shape
    with open('all_Xy_kmeans.pickle', 'wb') as file1:
        pickle.dump([X, hot], file1)