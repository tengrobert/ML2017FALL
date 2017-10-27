import numpy as np
import pandas as pd
import math
import sys

x_test = pd.read_csv(sys.argv[1])
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
    X = X / x_test_scale
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob



# fitted_values, cost_iter = grad_desc(np.zeros(x_train.shape[1]), x_train, y_train)
# fitted_values = pd.DataFrame(fitted_values)
# fitted_values.to_csv('./logistic_weight')
if __name__ == '__main__':
    fitted_values = [
    11.821495292363597
    ,1.2258314844628573
    ,0.8572697053552046
    ,27.17227984471244
    ,1.4832253409985214
    ,7.0247938605430456
    ,-1.0416078745769888
    ,-1.6898678282026964
    ,-0.5159818037801609
    ,-1.4717825548376524
    ,-1.2621350254964547
    ,-1.8901050538386044
    ,-1.8077645175111403
    ,-2.562147739485101
    ,-0.9915648005763843
    ,-1.5297798074429931
    ,-1.4192095518530334
    ,-1.0908123478144436
    ,-2.0440122840105563
    ,-1.7688787462038777
    ,-1.9288726789218542
    ,-1.7330692618674566
    ,-0.27526093464109064
    ,-0.26122584042693037
    ,0.3350144220419376
    ,1.4241010969724068
    ,-0.7698123347205151
    ,0.663425349187785
    ,-3.6425772789442927
    ,1.2338804564378503
    ,-0.4258674560980883
    ,-2.668015164390638
    ,0.42536715716513396
    ,-0.4323304757029261
    ,-2.6097371196297634
    ,-2.9443651373518516
    ,-2.7731150708856322
    ,-2.2307613875094563
    ,-0.6632114870087291
    ,-1.0115325882919959
    ,-0.6341571396749573
    ,0.10707775287839624
    ,-1.540603854492166
    ,-1.3179966664213214
    ,-0.9851167390857984
    ,-1.4505165386784635
    ,-2.9095173031590633
    ,-0.1490653042516413
    ,-0.022638081171126742
    ,-0.3679137078061068
    ,-0.012334881737552698
    ,-0.7678840550480606
    ,-1.5075466043565466
    ,-2.4582819819606088
    ,-1.8936495644193314
    ,-2.7727019466251486
    ,-2.953901983269803
    ,-2.0863204913584523
    ,-1.0681012306717848
    ,-2.9995120891091145
    ,-2.3377652673497944
    ,-2.6506868424463925
    ,-2.8046663048119527
    ,-2.4403266945879243
    ,0.9784386573626753
    ,0.21024263623486256
    ,-0.8590011327385294
    ,-2.0676058221475557
    ,0.2502470768803923
    ,-1.765427851569751
    ,-0.2519558915385453
    ,-0.7368912199988371
    ,0.25417842685859093
    ,0.45987097061471377
    ,0.3179791413196811
    ,-1.0086504823671059
    ,-0.3369471910857767
    ,-0.22976384917641238
    ,-0.03722714702175528
    ,-0.469223053593969
    ,-0.2130154950459908
    ,-0.07212330633579879
    ,-0.5278465036286052
    ,-0.1436928139342901
    ,0.516862239163279
    ,0.6287741738197818
    ,-0.10997739538460519
    ,0.2297401352077227
    ,-0.6372009546882562
    ,-0.7231316200528995
    ,-0.8984327160029435
    ,-1.2925764809520386
    ,-0.873223239678972
    ,0.2520853833218146
    ,-0.02594374819515869
    ,-0.1292611467881944
    ,-0.5286074376991515
    ,-0.17034892220893033
    ,-1.1833716087350716
    ,-0.06529144127284289
    ,-0.5458324030336571
    ,-0.5394757398051736
    ,0.07498064142639871
    ,-1.1786157410766052
    ,0.536368927680557
    ,-0.3220632524381791
    ,-11.657165029051262
    ,-0.5745173320907169
    ,3.6398064978717852
    ,2.729033991811474
    ,-4.6932517841850245
    ]
    predicted_y = pred_values(fitted_values, x_test)
    # print(predicted_y)

    output = pd.DataFrame(predicted_y)
    output.columns = ['label']
    output.index = [list(range(1, 16282))]
    output.index.name = 'id'
    output.to_csv(sys.argv[2])
# print(np.sum(y_train.squeeze() == predicted_y))


