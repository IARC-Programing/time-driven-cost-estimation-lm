alpha = 0.1


def leaky_relu(x):
    shape = x.shape
    x = x.flatten()
    result = 0
    if x > 0:
        result = x
    else:
        result = alpha * x
    return result.reshape(shape)


def leaky_relu_prime(x):
    x = x.flatten()
    if x > 0:
        return 1
    else:
        return alpha
