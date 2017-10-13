import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

def main():
    w = [1.41423778, -0.07356945, 0.08100093, 0.16498465, -0.25132885, -0.05843532, 0.67978913, -0.66656021, -0.12574716, 1.2299787]
    zzz = pd.read_csv(sys.argv[1])
    zzz.reset_index(inplace=True)
    zzz.drop(zzz.columns[[0,1,2]], axis=1, inplace=True)
    # testfeature1 = zzz.iloc[list(range(2,4319,18))]
    # testfeature1 = testfeature1.as_matrix()
    # testfeature2 = zzz.iloc[list(range(5,4319,18))]
    # testfeature2 = testfeature2.as_matrix()
    # testfeature3 = zzz.iloc[list(range(6,4319,18))]
    # testfeature3 = testfeature3.as_matrix()
    # testfeature4 = zzz.iloc[list(range(7,4319,18))]
    # testfeature4 = testfeature4.as_matrix()
    zzz = zzz.iloc[list(range(8,4319,18))]
    zzz = zzz.as_matrix()
    zzz = zzz[:,range(0,9)]
    # zzz = np.concatenate((testfeature1, testfeature2, testfeature3, testfeature4, zzz), axis=1)
    zzz = zzz.astype(float)
    #print(marks)
    # zzz = np.concatenate((zzz, zzz**2), axis=1)
    zzz = np.concatenate((np.ones((zzz.shape[0],1)), zzz), axis=1)
    #print(zzz)
    #print(zzz.dot(w))
    output = pd.DataFrame(zzz.dot(w).astype(int))
    output.columns = ['value']
    output.index = ["id_{}".format(x) for x in range(0,240)]
    output.index.name = 'id'
    output.to_csv(sys.argv[2])
if __name__ == "__main__":
    main()