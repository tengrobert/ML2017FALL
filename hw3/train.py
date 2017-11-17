import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

def read_data(filename, label=True, width=48, height=48):
    width = height = 48
    with open(filename, 'r') as f:
        data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
        data = np.array(data)
        X = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
        Y = data[::width*height+1].astype('int')

        X /= 255

        if label:
            return X, Y
        else:
            return X

X, Y = read_data('./train.csv')

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

input_shape = (48, 48, 1)
num_classes = y_test.shape[1]
batch_size = 128
epochs = 25

# Create the model
datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.8, 1.2],
        shear_range=0.2,
        horizontal_flip=True
)
model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), input_shape=input_shape, padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

# model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
# model.add(LeakyReLU(alpha=1./20))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.3))

# model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
# model.add(LeakyReLU(alpha=1./20))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.35))

# model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
# model.add(LeakyReLU(alpha=1./20))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.4))

model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('teng/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
    
model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=batch_size), 
        steps_per_epoch=5*len(X_train)//batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test)
)
model.save('my_model.h5')

# Compile model
# epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

test = read_data('./test.csv', label=False)
y_pred = model.predict(test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
output = pd.DataFrame(y_pred)
output.columns = ['label']
output.index = [list(range(0, 7178))]
output.index.name = 'id'
output.to_csv('./output.csv')