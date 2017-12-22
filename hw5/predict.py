import numpy as np
import pandas as pd
from keras.models import load_model
import sys

model = load_model('mfta-05-0.6228.h5')

test = pd.read_csv(sys.argv[1])
train = pd.read_csv('train.csv')
mean = 3.58
std = 1.11
y = model.predict([test['UserID'], test['MovieID']])
y = y * std + mean
output = pd.DataFrame(y)
output.columns = ['Rating']
output.index.name = 'TestDataID'
print(output.index.name)
output.index = list(range(1, y.shape[0] + 1))
output.Rating = output.Rating.round(1)
print(output)
output.to_csv(sys.argv[2],  index=True, index_label='TestDataID', float_format='%.1f')

