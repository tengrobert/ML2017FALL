import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from matplotlib import pyplot

word2vecmodel = Word2Vec.load('word2vec.bin')
vocab = dict([(k, v.index) for k,v in word2vecmodel.wv.vocab.items()])
embedding_matrix = word2vecmodel.wv.syn0
removed_punctuation = '\'+$,0123456789'

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
        print('------read labeled data------')
        data = f.read()
        data = data.splitlines(False)
        Y = [int(line[0]) for line in data]
        print(Y[:5])
        table = str.maketrans('', '', string.punctuation)
        data = [line.translate(table) for line in data]
        print(data[:2])
        #data = [line[1:] for line in data]
        #print(data[0])
        data = [line.split() for line in data]
        print(data[:2])
        data = [convert_data_to_index(line) for line in data] 
        print(data[:2])

        if label:
            return data, Y
        else:
            return X

def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        print('------read nolabel data------')
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', string.punctuation)
        data = [line.translate(table) for line in data]
        nolabel_fitted = pd.read_csv('./nolabel_fitted.csv')
        fitted_idx = nolabel_fitted['id'].tolist()
        X = [data[i] for i in fitted_idx]
        print(X[:3])
        X = [line.split() for line in X]
        print(X[:3])
        X = [convert_data_to_index(line) for line in X]
        print(X[:3])
        Y = nolabel_fitted['label'].tolist()
        print(Y[:3])
        return X, Y

def read_nolabel_data_prediction(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data=[line.translate(table) for line in data]
        data=[line.split() for line in data]
        print(data[:3])
        data = [convert_data_to_index(line) for line in data]
        print(data[:3])
        return data

X, Y = read_data('../training_label.txt')
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
# load the model
modelname = 'without_number_gensim-23-0.8219.h5'#'weights-improvement-06-0.80.h5'
model = load_model(modelname)

# checkpoint
filepath="semigensim-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

for i in range(10):
    #predict
    print("#########Iteration %d#########" % (i))
    nolabel = read_nolabel_data_prediction('../training_nolabel.txt')
    nolabel = sequence.pad_sequences(nolabel, maxlen=max_review_length)
    prediction = model.predict(nolabel, batch_size=1024, verbose=True)
    output = pd.DataFrame(prediction)
    output.columns = ['label']
    output.index.name = 'id'
    output.index = list(range(0, prediction.shape[0]))
    output.to_csv('./nolabel_prediction.csv', encoding='utf-8', index=True, index_label='id')
    #take what we want avvording to threshold and round the label
    Y_nolabel = pd.read_csv('./nolabel_prediction.csv')
    threshold = 0.01*i
    # if i > 9:
    #     threshold = 0.09 + 0.001*(i - 9)
    Y_nolabel = Y_nolabel[(Y_nolabel['label'] > 0.9) | (Y_nolabel['label'] < 0.1)]
    Y_nolabel = Y_nolabel.round().astype(int)
    Y_nolabel.to_csv('nolabel_fitted.csv', index=False)
    #semi-supervised
    X_nolabel, Y_nolabel = read_nolabel_data('../training_nolabel.txt')
    X_nolabel = sequence.pad_sequences(X_nolabel, maxlen=max_review_length)
    X = np.concatenate((X_train, X_nolabel), axis=0)
    y = np.concatenate((y_train, Y_nolabel), axis=0)
    myepoch = 2
    if i == 9:
        myepoch = 50
    history = model.fit(X, y, epochs=myepoch, validation_data=(X_test, y_test), batch_size=256, callbacks=callbacks_list)
    #load new model
    model = load_model('semigensim-00.h5')
