import numpy as np
import pandas as pd
import string
import sys
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import plot_model

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

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data =[line.translate(table) for line in data]
        data.pop(0)
        print(data[:3])
        #data = [line[1:] for line in data]
        #print(data[:3])
        data = [line.split() for line in data]
        print(data[:3])
        data = [convert_data_to_index(line) for line in data] 
        print(data[:3])

       
        return data
X_test = read_data(sys.argv[1])
print(X_test[:10])
modelname = './semigensim-0.8189onkaggle.h5'
model = load_model(modelname)
# plot_model(model, to_file='RNN.png')
max_review_length = 36
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
prediction = model.predict(X_test, verbose=1)
print(prediction[:5])
prediction = np.where(prediction >= .5, 1, 0)
output = pd.DataFrame(prediction)
output.columns = ['label']
output.index.name = 'id'
print(output.index.name)
output.index = list(range(0, 200000))
print(output)
output.to_csv(sys.argv[2], encoding='utf-8', index=True, index_label='id')
