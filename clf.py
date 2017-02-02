from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt

import sys
import os
import pickle
import numpy as np

bufferpath = "../../python/signalProc"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), bufferpath))

# read data
with open('all_Xy_kmeans.pickle', 'rb') as file:
    X, y = pickle.load(file)
# with open('Vicky_newdata.pickle', 'rb') as file:
#     X = pickle.load(file)
# with open('Vicky_labels.pickle', 'rb') as file:
#     y = pickle.load(file)
#     y = y['hot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print X_train.shape
X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)
# X_train = np.expand_dims(np.array(X_train), axis=-1)
# X_test = np.expand_dims(np.array(X_test), axis=-1)
y_train = np.array(y_train)
y_test = np.array(y_test)

# relevant variables
# input_size = (X_train.shape[1], X_train.shape[2])
input_size = X_train.shape[1:]
output_size = y_train.shape[1]

print X_train.shape, y_train.shape

# print input_size, output_size

clfs = []
hist = []

# for i in range(output_size):
#
#     idx = np.where(y_train[:, i] == 1)
#
#     y = [1] * len(idx[0])
#     X = X_train[idx[0]]
#
#     # model
#     model = Sequential()
#     model.add(Dense(10, input_shape=input_size))
#     # model.add(LSTM(10, input_shape=input_size))
#     # model.add(Dense(1000, activation='relu'))
#     # model.add(Dense(500, activation='relu'))
#     model.add(Dense(10, activation='tanh'))
#     # model.add(Dense(100, activation='tanh'))
#     # model.add(Dense(250, activation='relu'))
#     # model.add(Dense(1000, activation='relu'))
#     model.add(Dense(output_dim=1, activation='sigmoid'))
#     # compile model
#     model.compile(optimizer='rmsprop',
#       loss='binary_crossentropy',
#       metrics=['accuracy'])
#
#     #train
#     history = model.fit(X, y, batch_size=32, nb_epoch=50, verbose=1, validation_split=0.1, shuffle=False)
#     clfs.append(model)
#     hist.append(history)
#
# predictions = []
# for c in clfs:
#     preds = c.predict(X_test, verbose=1)
#     predictions.append(preds)
#
# ind = np.argmax(predictions, axis=0)
#
# hit = 0
#
# for i, t in zip(ind, y_test):
#     if t[i] == 1:
#         hit += 1
#
# acc = float(hit) / len(y_test)
# print '\n accuracy ', acc
#

# # model
# model = Sequential()
# model.add(Embedding(max_features, 256, input_length=maxlen))
# model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
# score = model.evaluate(X_test, Y_test, batch_size=16)
from keras.layers import Convolution2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Dense(100, input_shape=input_size))
model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
# model.add(Dropout(0.1))
model.add(Dense(100, activation='tanh'))
model.add(Dense(50, activation='tanh'))
# model.add(Dropout(0.1))
model.add(Dense(output_dim=output_size, activation='softmax'))

# model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu', input_shape=input_size))
# # model.add(Convolution2D(64, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))


# compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit(X_train, y_train, batch_size=32, nb_epoch=200, verbose=1, validation_split=0.1, shuffle=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
figloss = plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

figacc = plt.figure(1)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# preds = c.predict(X_test, verbose=1)
preds = model.predict_classes(X_test, verbose=1)

true = []
for x in y_test:
    i = np.where(x)[0][0]
    true.append(i)

for i in range(len(true)):

    if true[i] > 2:
        true[i] = 3

    if preds[i] > 2:
        preds[i] = 3

print true, preds
cm = confusion_matrix(true, preds)

from sklearn import preprocessing

cm = preprocessing.normalize(cm.astype('float32'), axis=1)

plt.figure(2)
plt.matshow(cm)
plt.xticks(np.arange(4), ['R-H', 'L-H', 'Feet', 'Rest'])
plt.yticks(np.arange(4), ['R-H', 'L-H', 'Feet', 'Rest'])
plt.xlabel('True')
plt.ylabel('Predcited')
plt.colorbar()
plt.show()

# ind = np.argmax(predictions, axis=0)
#
# hit = 0
#
# for i, t in zip(ind, y_test):
#     if t[i] == 1:
#         hit += 1
#
# acc = float(hit) / len(y_test)
# print '\n accuracy ', acc
