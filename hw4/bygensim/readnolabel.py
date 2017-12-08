import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.preprocessing import sequence

word2vecmodel = Word2Vec.load('word2vec.bin')
vocab = dict([(k, v.index) for k,v in word2vecmodel.wv.vocab.items()])
removed_punctuation = '\'+$,0123456789'

def convert_data_to_index(words):
    def word_to_id(word):
        id = vocab.get(word)
        if id is None:
            id = 0
        return id

    words = list(map(word_to_id, words))
    return words

def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data = [line.translate(table) for line in data]
        data = [line.split() for line in data]
        print(data[:3])
        data = [convert_data_to_index(line) for line in data]
        print(data[:3])
        return data

nolabel = read_nolabel_data('../training_nolabel.txt')
Y_nolabel = pd.read_csv('./nolabel_prediction.csv')
Y_nolabel = Y_nolabel[(Y_nolabel['label'] > 0.9) | (Y_nolabel['label'] < 0.1)]
Y_nolabel = Y_nolabel.round().astype(int)
Y_nolabel.to_csv('nolabel_fitted.csv', index=False)