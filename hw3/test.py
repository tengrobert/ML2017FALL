import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model


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


# Fit the model
model = load_model('mymodel.h5')
test = read_data(sys.argv[1], label=False)
y_pred = model.predict(test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
output = pd.DataFrame(y_pred)
output.columns = ['label']
output.index = [list(range(0, 7178))]
output.index.name = 'id'
output.to_csv(sys.argv[2])