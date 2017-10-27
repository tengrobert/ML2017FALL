import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation



x_test = pd.read_csv('./X_test')
x_train = pd.read_csv('./X_train')
y_train = pd.read_csv('./Y_train')
x_test = x_test.as_matrix()
x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_train = np.concatenate((x_train, np.square(x_train[:,[0]]), np.square(x_train[:,[1]]), np.square(x_train[:,[3]]), np.square(x_train[:,[4]]), np.square(x_train[:,[5]])), axis=1)
x_test = np.concatenate((x_test, np.square(x_test[:,[0]]), np.square(x_test[:,[1]]), np.square(x_test[:,[3]]), np.square(x_test[:,[4]]), np.square(x_test[:,[5]])), axis=1)
x_train_scale = x_train.max(axis=0) - x_train.min(axis=0)
x_train_scale = np.where(x_train_scale != 0, x_train_scale, 1)
x_test_scale = x_test.max(axis=0) - x_test.min(axis=0)
x_test_scale = np.where(x_test_scale != 0, x_test_scale, 1)

X_train = x_train / x_train_scale
X_test = x_test / x_test_scale


input_dim = X_train.shape[1]
batch_size = 80
nb_epoch = 300

Y_train = y_train

model = Sequential()
model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))

model.summary()

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_train, Y_train))
score = model.evaluate(X_train, Y_train, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('my_model.h5')

result = model.predict(X_test)
pred_value = np.where(result >= .5, 1, 0)
output = pd.DataFrame(pred_value)
output.columns = ['label']
output.index = [list(range(1, 16282))]
output.index.name = 'id'
output.to_csv('./output.csv')