# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# load data
x_test = pd.read_csv('./X_test')
x_train = pd.read_csv('./X_train')
y_train = pd.read_csv('./Y_train')
x_test = x_test.as_matrix()
x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
y_train = y_train.squeeze()
x_train = np.concatenate((x_train, x_train[:,[5]]**2), axis=1)
x_test = np.concatenate((x_test, np.square(x_test[:,[5]])), axis=1)
x_train_scale = x_train.max(axis=0) - x_train.min(axis=0)
x_train_scale = np.where(x_train_scale != 0, x_train_scale, 1)
x_test_scale = x_test.max(axis=0) - x_test.min(axis=0)
x_test_scale = np.where(x_test_scale != 0, x_test_scale, 1)
#preprocess
scaler = preprocessing.MinMaxScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
scaler.fit(x_test)
X_test = scaler.transform(x_test)
# X_train = preprocessing.normalize(x_train, axis=0, norm='max')
# X_train = preprocessing.scale(X_train, axis=0)
# X_test = preprocessing.normalize(x_test, axis=0, norm='max')
# X_test = preprocessing.scale(X_test, axis=0)
# seed = 7
# test_size = 0.33
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier(max_depth=5)
model.fit(X_train, y_train)

# make predictions for test data
# y_pred = model.predict(Xtest)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# # print(np.sum(y_train == y_pred))
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_train, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
output = pd.DataFrame(y_pred)
output.columns = ['label']
output.index = [list(range(1, 16282))]
output.index.name = 'id'
output.to_csv('./xgboutput.csv')