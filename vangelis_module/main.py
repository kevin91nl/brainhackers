import utils
from utils.data import *
from utils.processing import *
from utils.common import *
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed, Dropout, Activation, Embedding, Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from sklearn.metrics import accuracy_score




vocab_size=40
labels=[]
for vocab_size in [1,5,10,20,30,40,50]:
    labels.append("Vocab: %d"%vocab_size)
    corpus=dataset()
    processor_=processor(vocab_size=vocab_size)

    print(corpus.X.shape)
    print(corpus.Frames.shape)


    #training Kmeans
    processor_.kmeans=corpus.Frames


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
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()




    #prepare data
    data_x=np.zeros((1620,19))
    for batch in range(corpus.X.shape[0]):
        data_x[batch,:]=processor_.kmeans.predict(corpus.X[batch, :, :])




    #1000 for training
    history = model.fit(data_x[:1620], corpus.Y[:1620], batch_size=32, nb_epoch=20, validation_split=.2)


    # plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')


plt.legend(labels, loc='upper left')
plt.show()



