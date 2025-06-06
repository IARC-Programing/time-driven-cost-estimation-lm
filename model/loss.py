import numpy as np

# lost function and its derivatives


def mse(y_true, y_pred):
    # print(f'y-true = {y_true} & y-predict = {y_pred}')
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    # dE/dY = d(ERROR ROOT MEAN SQUARE)/dY
    # dE/dY = d/dy (y-y_predict)^2
    # dE/dY = 2(y-y_predict)
    # dE/dY = 2*error
    return 2*(y_pred-y_true)/y_true.size


def rmspe(y_true, y_pred):
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    return rmspe
