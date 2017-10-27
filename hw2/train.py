import numpy as np
import pandas as pd
import math

x_test = pd.read_csv('./X_test')
x_train = pd.read_csv('./X_train')
y_train = pd.read_csv('./Y_train')
x_test = x_test.as_matrix()
x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_train_max = x_train.max(axis=0)
x_train = np.concatenate((x_train, np.square(x_train[:,[0]]), np.square(x_train[:,[1]]), np.square(x_train[:,[3]]), np.square(x_train[:,[4]]), np.square(x_train[:,[5]])), axis=1)
x_test = np.concatenate((x_test, np.square(x_test[:,[0]]), np.square(x_test[:,[1]]), np.square(x_test[:,[3]]), np.square(x_test[:,[4]]), np.square(x_test[:,[5]])), axis=1)
x_train_scale = x_train.max(axis=0) - x_train.min(axis=0)
x_train_scale = np.where(x_train_scale != 0, x_train_scale, 1)
x_test_scale = x_test.max(axis=0) - x_test.min(axis=0)
x_test_scale = np.where(x_test_scale != 0, x_test_scale, 1)


def logistic_func(theta, x):
    return float(1) / (1 + math.e**(-x.dot(theta)))

def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - y.squeeze()
    final_calc = first_calc.T.dot(x)
    return final_calc

def cost_func(theta, x, y):
    log_func_v = logistic_func(theta, x)
    y = np.squeeze(y)
    # J = (1./x.shape[0]) * (-y.T.dot(np.log(log_func_v)) - np.transpose(1-y).dot(np.log(1-log_func_v)))
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

def grad_desc(theta_values, X, y, lr=2, converge_change=.0000001):
    #normalize
    X = X / x_train_scale
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y)) / X.shape[0]
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        print("Iteration %d | Cost: %f" % (i, cost))
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)

def pred_values(theta, X, hard=True):
    X = X / x_train_scale
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob



fitted_values, cost_iter = grad_desc(np.zeros(x_train.shape[1]), x_train, y_train)
predicted_y = pred_values(fitted_values, x_train)
# print(predicted_y)

# output = pd.DataFrame(predicted_y)
# output.columns = ['label']
# output.index = [list(range(1, 16282))]
# output.index.name = 'id'
# output.to_csv('./outputbasic.csv')
print(np.sum(y_train.squeeze() == predicted_y))


