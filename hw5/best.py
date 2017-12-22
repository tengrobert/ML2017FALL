import pandas as pd
import numpy as np
import pickle
import sys

test = pd.read_csv(sys.argv[1])
m = 0
with open('m.pickle', 'rb') as handle:
    m = pickle.load(handle)

test['Rating'] = test.apply(lambda x:m[m.index==x.UserID][x.MovieID].values[0], axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.Rating.isnull())[0]
test.ix[missing,'Rating'] = 3.78
test.Rating = test.Rating.round(1)

test.to_csv(sys.argv[2],index=False,columns=['TestDataID','Rating'])