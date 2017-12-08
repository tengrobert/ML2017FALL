import string
import numpy as np
from gensim.models import Word2Vec

removed_punctuation = '\'+$,0123456789'

def read_data(filename, label=True):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        Y = [int(line[0]) for line in data]
        table = str.maketrans('', '', removed_punctuation)
        print(data[:10])
        data=[line.translate(table) for line in data]
        print(data[:10])
        #data = [line[1:] for line in data]
        X = [line.split() for line in data]
        print(X[:5])

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
        print(data[:5])
        #data = [line[1:] for line in data]
        data = [line.split() for line in data]
        return data

def read_nolabel_data(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read()
        data = data.splitlines(False)
        table = str.maketrans('', '', removed_punctuation)
        data=[line.translate(table) for line in data]
        data=[line.split() for line in data]
        return data

X, Y = read_data('../training_label.txt')
X_test = read_test_data('../testing_data.txt')
nolabel = read_nolabel_data('../training_nolabel.txt')


result = np.concatenate((X, X_test, nolabel), axis=0)

model = Word2Vec(result, window=5, size=100, min_count=170)
print(model)
# save model
model.save('word2vec.bin')
