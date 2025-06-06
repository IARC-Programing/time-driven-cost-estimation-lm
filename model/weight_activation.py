import numpy as np

alpha = 0


def un_zero_weight(x):
    shape = x.shape
    x = x.flatten()
    result = np.where(x < 0, 0, x)
    return result.reshape(shape)
