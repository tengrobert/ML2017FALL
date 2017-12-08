import numpy as np
import pandas as pd
import string
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pickle

t = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)

def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', string.punctuation)
        data=[line.translate(table) for line in data]
        data=t.texts_to_sequences(data)
        return data

nolabel = read_nolabel_data('training_nolabel.txt')
Y_nolabel = pd.read_csv('./nolabel_prediction.csv')
Y_nolabel = Y_nolabel[(Y_nolabel['label'] > 0.9) | (Y_nolabel['label'] < 0.1)]
Y_nolabel = Y_nolabel.round().astype(int)
Y_nolabel.to_csv('nolabel_fitted.csv', index=False)