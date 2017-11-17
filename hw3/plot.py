import pandas as pd
import numpy as np
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
emotion_classifier = load_model('zzzmodel-00001-0.26409.h5')
emotion_classifier.summary()
# plot_model(emotion_classifier,to_file='model.png')

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

# seed = 7
# test_size = 0.1
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# y_test_categories = y_test
# y_test = np_utils.to_categorical(y_test)

# prediction = emotion_classifier.predict_classes(X_test)

# print(y_test.shape)

# print(pd.crosstab(y_test_categories, prediction, rownames=['label'], colnames=['predict']))

