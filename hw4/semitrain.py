import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from matplotlib import pyplot
import pickle

#word2vecmodel = Word2Vec.load('word2vec.bin')
#embedding_matrix = word2vecmodel[word2vecmodel.wv.vocab]
t = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', string.punctuation)
        data=[line.translate(table) for line in data]
        print(data[:2])
        Y = [int(line[0]) for line in data]
        print(Y[:5])
        data = [line[1:] for line in data]
        print(data[0])
        data=t.texts_to_sequences(data)
        print(data[:2])

        if label:
            return data, Y
        else:
            return X

def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', string.punctuation)
        data=[line.translate(table) for line in data]
        nolabel_fitted = pd.read_csv('./nolabel_fitted.csv')
        fitted_idx = nolabel_fitted['id'].tolist()
        X = [data[i] for i in fitted_idx]
        print(X[:3])
        X = t.texts_to_sequences(X)
        print(X[:3])
        Y = nolabel_fitted['label'].tolist()
        print(Y[:3])
        return X, Y

X, Y = read_data('training_label.txt')
X_nolabel, Y_nolabel = read_nolabel_data('./training_nolabel.txt')
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
seed = 7
test_size = 0.05
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

X_train = np.concatenate((X_train, X_nolabel), axis=0)
y_train = np.concatenate((y_train, Y_nolabel), axis=0)
# truncate and pad input sequences
max_review_length = 36
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 48
model = Sequential()
model.add(Embedding(5000, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
#model.add(Dropout(0.2))
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()
# checkpoint
filepath="semi-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=256, callbacks=callbacks_list)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1, batch_size=256)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('mymodel.h5')
