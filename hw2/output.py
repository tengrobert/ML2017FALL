import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model

model = load_model('my_model.h5')

x_test = pd.read_csv('./X_test')
x_train = pd.read_csv('./X_train')
y_train = pd.read_csv('./Y_train')
x_test = x_test.as_matrix()
x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_train_max = x_train.max(axis=0)

X_train = x_train / x_train_max
X_test = x_test / x_train_max


result = model.predict(X_test)
print(result)
pred_value = np.where(result >= .5, 1, 0)
output = pd.DataFrame(pred_value)
output.columns = ['label']
output.index = [list(range(1, 16282))]
output.index.name = 'id'
output.to_csv('./output.csv')