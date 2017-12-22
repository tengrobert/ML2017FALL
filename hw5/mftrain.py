import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import random
import pickle

pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
matrix = pd.concat([train,test]).pivot('UserID','MovieID','Rating')
movie_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-movie_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999
while iteration<40:
    iteration += 1
    svd = TruncatedSVD(n_components=15,random_state=40)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+movie_means
m = m.clip(lower=1,upper=5)
with open('m.pickle', 'wb') as handle:
    pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('m save')

test['Rating'] = test.apply(lambda x:m[m.index==x.UserID][x.MovieID].values[0], axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.Rating.isnull())[0]
test.ix[missing,'Rating'] = user_means[test.loc[missing].UserID].values
test.Rating = test.Rating.round(1)

test.to_csv('submission.csv',index=False,columns=['TestDataID','Rating'])