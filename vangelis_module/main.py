import utils
from utils.data import *
from utils.processing import *
from utils.common import *
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed, Dropout, Activation, Embedding, Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from sklearn.metrics import accuracy_score

vocab_size=30

corpus=dataset()
processor=processor(vocab_size=vocab_size)

print(corpus.X.shape)
print(corpus.Frames.shape)

#training Kmeans
processor.kmeans=corpus.Frames


#model Embedding LSTM
embedding_dims=embedding_dim()

model = Sequential()

model.add(Embedding(vocab_size, embedding_dims,
                    dropout=.0,trainable=True))


# model.add(Convolution1D(nb_filter=64,
#                             filter_length=5,
#                             border_mode='valid',
#                             activation='relu',
#                             subsample_length=1))
# model.add(MaxPooling1D(pool_length=4))


model.add(LSTM(128, dropout_W=.2, dropout_U=.2, return_sequences=True))
model.add(TimeDistributed(Dense(4)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




model.summary()
#prepare data
data_x=np.zeros((1620,20))
for batch in range(corpus.X.shape[0]):
    data_x[batch,:]=processor.kmeans.predict(corpus.X[batch,:,:])

#1000 for training
model.fit(data_x[:1000], corpus.Y[:1000], batch_size=32, nb_epoch=2, validation_split=.0)

y_output = model.predict(data_x)
y_output[np.where(y_output<0.5)]=0
y_output[np.where(y_output>=0.5)]=1

#620 for testing
accuracy=0
for i in range(1000,1620):
    for j in range(20):
        accuracy += accuracy_score(y_output[i,j], corpus.Y[i,j])
print("Mean Accuracy: %f"%(accuracy/(620*20)))


