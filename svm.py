from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import sys
import os
import pickle
import numpy as np

bufferpath = "../../python/signalProc"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), bufferpath))
import bufhelp


# read data
with open('all_Xy_kmeans.pickle', 'rb') as file:
    X, y = pickle.load(file)
# with open('Monica_newdata.pickle', 'rb') as file:
#     X = pickle.load(file)
# with open('Monica_labels.pickle', 'rb') as file:
#     y = pickle.load(file)
#     y = y['hot']

print X.shape, y.shape
scores = []
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_train = np.array(X_train).reshape(len(X_train), -1)
    X_test = np.array(X_test).reshape(len(X_test), -1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = np.array(np.where(y_train)[1])
    y_test = np.array(np.where(y_test)[1])

    clf = SVC()
    clf.fit(X_train, y_train)
    SVC(C=1.0, cache_size=200, kernel='rbf',
        tol=0.001, verbose=True)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print score
    scores.append(score)
print 'mean', np.mean(scores)
print 'std', np.std(scores)


for i in range(len(y_test)):

    if y_test[i] > 2:
        y_test[i] = 3

    if y_pred[i] > 2:
        y_pred[i] = 3

cm = confusion_matrix(y_test, y_pred)

from sklearn import preprocessing
cm = preprocessing.normalize(cm.astype('float32'), axis=1)

plt.figure(2)
plt.matshow(cm)
plt.xticks(np.arange(4), ['R-H', 'L-H', 'Feet', 'Rest'])
plt.yticks(np.arange(4), ['R-H', 'L-H', 'Feet', 'Rest'])
# plt.xticks(np.arange(4), ['Rest', 'R-H', 'L-H', 'Feet'])
# plt.yticks(np.arange(4), ['Rest', 'R-H', 'L-H', 'Feet'])
plt.xlabel('True')
plt.ylabel('Predcited')
plt.colorbar()
plt.show()
