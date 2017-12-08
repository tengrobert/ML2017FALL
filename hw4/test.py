import numpy as np
import pandas as pd
import string
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
import pickle

#word2vecmodel = Word2Vec.load('word2vec.bin')
t = Tokenizer()

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', string.punctuation)
        data=[line.translate(table) for line in data]
        data.pop(0)
        print(data[:3])
        data = [line[1:] for line in data]
        print(data[:3])
        with open('tokenizer.pickle', 'rb') as handle:
            t = pickle.load(handle)
        data=t.texts_to_matrix(data)
        print(data[:3])
        #data=[line[0].split() for line in data]
        #X = [convert_data_to_index(line, word2vecmodel.wv) for line in data]
        #Y = [line[0] for line in data]

       
        return data
X_test = read_data('./testing_data.txt')
print(X_test[:10])
modelname = './BOW-weights-04-0.80.h5'
model = load_model(modelname)
plot_model(model, to_file='BOW.png')
print('model plotted')
# max_review_length = 36
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
prediction = model.predict(X_test)
print(prediction[:5])
prediction = np.where(prediction >= .5, 1, 0)
output = pd.DataFrame(prediction)
output.columns = ['label']
output.index.name = 'id'
print(output.index.name)
output.index = list(range(0, 200000))
print(output)
output.to_csv('./BOW.csv', encoding='utf-8', index=True, index_label='id')
