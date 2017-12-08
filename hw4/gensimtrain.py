import numpy as np
import string
import sys
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from matplotlib import pyplot
import pickle

word2vecmodel = Word2Vec.load('word2vec.bin')
embedding_matrix = word2vecmodel.wv.syn0
vocab = dict([(k, v.index) for k,v in word2vecmodel.wv.vocab.items()])
removed_punctuation = '\'+$,0123456789'

print(embedding_matrix.shape)


def convert_data_to_index(words):
    def word_to_id(word):
        id = vocab.get(word)
        if id is None:
            id = 0
        return id

    words = list(map(word_to_id, words))
    return words

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        Y = [int(line[0]) for line in data]
        table = str.maketrans('', '', removed_punctuation)
        data = [line.translate(table) for line in data]
        print(Y[:5])
        #data = [line[1:] for line in data]
        data = [line.split() for line in data]
        print(data[:3])
        #t.fit_on_texts(data)
        data=[convert_data_to_index(line) for line in data]
        print(data[:3])
        #X = [line[1:len(line)] for line in data]
        #X = [convert_data_to_index(line, word2vecmodel.wv) for line in data]
        #Y = [int(line[0]) for line in data]

        if label:
            return data, Y
        else:
            return X

X, Y = read_data(sys.argv[1])
#vocab_size = len(t.word_index) + 1
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print('Tokenizer save')
#print(Y)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
seed = 7
test_size = 0.05
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# truncate and pad input sequences
max_review_length = 36
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, input_length=max_review_length))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
#model.add(Dropout(0.2))
model.add(LSTM(48, dropout=0.4, recurrent_dropout=0.4))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()
# checkpoint
filepath="gensim-{epoch:02d}-{val_acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=256, callbacks=callbacks_list)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1, batch_size=256)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('mymodel.h5')
