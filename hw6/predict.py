import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from keras.layers import Input, Dense
from keras.models import load_model
import math
import h5py
import sys

encoder = load_model('./encoder.h5')
train = np.load(sys.argv[1])
test = pd.read_csv(sys.argv[2])

feature = encoder.predict(train)

kmeans = KMeans(n_clusters=2,random_state=0).fit(feature)

ans = []
for index, row in test.iterrows():
    ans.append(int(kmeans.labels_[row['image1_index']] == kmeans.labels_[row['image2_index']]))
    if(index%100000==0):
        print(index)

test['Ans'] = pd.DataFrame(ans)
test.to_csv(sys.argv[3],index=False,columns=['ID','Ans'])