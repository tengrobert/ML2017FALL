import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.models import load_model
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
modelname = 'best-31-0.81.h5'#'weights-improvement-06-0.80.h5'
model = load_model(modelname)
max_review_length = 36
nolabel = sequence.pad_sequences(nolabel, maxlen=max_review_length)
prediction = model.predict(nolabel)
print(prediction[:5])
output = pd.DataFrame(prediction)
output.columns = ['label']
output.index.name = 'id'
print(output.index.name)
output.index = list(range(0, prediction.shape[0]))
print(output)
output.to_csv('./nolabel_prediction.csv', encoding='utf-8', index=True, index_label='id')
