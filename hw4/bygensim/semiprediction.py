import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.models import load_model
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
        data=[line.translate(table) for line in data]
        data=[line.split() for line in data]
        print(data[:3])
        data = [convert_data_to_index(line) for line in data]
        print(data[:3])
        return data

nolabel = read_nolabel_data('../training_nolabel.txt')
modelname = 'without_number_gensim-23-0.8219.h5'#'weights-improvement-06-0.80.h5'
model = load_model(modelname)
max_review_length = 36
nolabel = sequence.pad_sequences(nolabel, maxlen=max_review_length)
prediction = model.predict(nolabel, batch_size=1024, verbose=True)
print(prediction[:5])
output = pd.DataFrame(prediction)
output.columns = ['label']
output.index.name = 'id'
print(output.index.name)
output.index = list(range(0, prediction.shape[0]))
print(output)
output.to_csv('./nolabel_prediction.csv', encoding='utf-8', index=True, index_label='id')
