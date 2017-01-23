import utils
from utils.data import *
from utils.processing import *
from utils.common import *
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed, Dropout, Activation, Embedding, Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from tqdm import tqdm

vocab_size=30

corpus=dataset()
processor=processor(vocab_size=vocab_size)

print(corpus.X.shape)
print(corpus.Frames.shape)

#training Kmeans
processor.kmeans=corpus.Frames


#model Embedding CNN LSTM
embedding_dims=embedding_dim()

model = Sequential()

model.add(Embedding(vocab_size, embedding_dims,
                    dropout=.0,trainable=True))
model.add(LSTM(128, dropout_W=.0, dropout_U=.0, return_sequences=True))
model.add(TimeDistributed(Dense(4)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])





#prepare data
data_x=np.zeros((1620,20))
for batch in range(corpus.X.shape[0]):
    data_x[batch,:]=processor.kmeans.predict(corpus.X[batch,:,:])

model.fit(data_x, corpus.Y, batch_size=32, nb_epoch=2, validation_split=.1)





