import string
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
import pickle
t = Tokenizer(num_words=5000)
removed_punctuation = '\'+$,0123456789'

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        Y = [int(line[0]) for line in data]
        table = str.maketrans('', '', removed_punctuation)
        X = [line.translate(table) for line in data]
        #data=[line[0].split() for line in data]
        #X = [line[1:] for line in data]

        if label:
            return X, Y
        else:
            return X
def read_test_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data=[line.translate(table) for line in data]
        data.pop(0)
        data = [line[1:] for line in data]
        #X = [convert_data_to_index(line, word2vecmodel.wv) for line in data]
        return data
def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data=[line.translate(table) for line in data]
        #data=[line[0].split() for line in data]
        return data

X_test = read_test_data('testing_data.txt')
nolabel = read_nolabel_data('training_nolabel.txt')
X, Y = read_data('training_label.txt')

print(X[:3])
print(nolabel[:3])

result = np.concatenate((X, X_test, nolabel), axis=0)
t.fit_on_texts(result)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Tokenizer save')
# model = Word2Vec(result, window=3, size=100, min_count=30)
# print(model)
# # save model
# model.save('word2vec.bin')
