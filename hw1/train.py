import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def main():
    marks = pd.read_csv('./train.csv')
    marks.reset_index(inplace=True)
    marks.drop('日期', 1, inplace=True)
    marks.drop('測站', 1, inplace=True)
    marks.drop('測項', 1, inplace=True)
    marks.drop('index', 1, inplace=True)
    #print(marks['0'])
    def gradientDescent(x, y, theta, alpha, m, s_gra, numIterations):
        xTrans = x.transpose()
        for i in range(0, numIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y + 0.1 * np.sum(theta**2)
            # avg cost per example (the 2 in 2*m doesn't really matter here.
            # But to be consistent with the gradient, I include it)
            cost = np.sum(loss ** 2)/ m
            cost_a = math.sqrt(cost)
            print("Iteration %d | Cost: %f" % (i, cost_a))
            # avg gradient per example
            gradient = np.dot(xTrans, loss)
            s_gra += gradient**2
            ada = np.sqrt(s_gra)
            # update
            theta = theta - alpha * gradient / ada
        return theta


    # feature1 = marks.iloc[list(range(3,4320,18))]
    # feature1 = feature1.as_matrix()
    # feature1 = feature1[:,list(range(0,9))]

    # feature2 = marks.iloc[list(range(6,4320,18))]
    # feature2 = feature2.as_matrix()
    # feature2 = feature2[:,list(range(0,9))]

    # feature3 = marks.iloc[list(range(7,4320,18))]
    # feature3 = feature3.as_matrix()
    # feature3 = feature3[:,list(range(0,9))]

    # feature4 = marks.iloc[list(range(8,4320,18))]
    # feature4 = feature4.as_matrix()
    # feature4 = feature4[:,list(range(0,9))]

    marks = marks.iloc[list(range(9,4320,18))]
    marks = marks.as_matrix()
    X = marks[:,list(range(0,9))]
    # X = np.concatenate((feature1, feature2, feature3, feature4, X), axis=1)

    # X = np.concatenate((X, X**2), axis=1)
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    # amin = np.min(X, axis=1)
    # amax = np.max(X, axis=1)
    # mins = amin
    # maxs = amax
    # for i in range(9):
    #     mins = np.c_[mins, amin]
    #     maxs = np.c_[maxs, amax]

    # X = (X - mins) / (maxs - mins)
    # X = X[~np.isnan(X).any(axis=1)]

    # print(X)

    # amean = np.mean(X, axis=1)
    # astd = np.std(X, axis=1)
    # means = amean
    # stds = astd
    # for i in range(9):
    #     means = np.c_[means, amean]
    #     stds = np.c_[stds, astd]
    # X = (X - means) / stds

    # amin = np.min(marks, axis=1)
    # amax = np.max(marks, axis=1)
    # mins = amin
    # maxs = amax
    # for i in range(23):
    #     mins = np.c_[mins, amin]
    #     maxs = np.c_[maxs, amax]

    # marks = (marks - mins) / (maxs - mins)


    w = gradientDescent(X, marks[:,9], np.zeros(X.shape[1]), 10, X.shape[0], np.zeros(X.shape[1]), 50000 )
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
    print(w)
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