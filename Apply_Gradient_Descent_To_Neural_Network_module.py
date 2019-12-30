"""
Neural Network Basic necessary module
"""


def _cross_entropy(y_pred, y_true, delta=1e-7):
    # np.log(0) returns RuntimeWarning: divide by zero encountered in log error, prints -inf
	
    return -np.sum(y_true * np.log(y_pred + delta))
	
	"""
	1. since log function is not defined at x = 0, padding 
	"""


def cross_entropy(y_pred, y_true, delta=1e-7):
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true, delta)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true, delta) / len(y_pred)
    return ce
	
def _numerical_gradient(fn, x):
    """
	suppose fn be multi variable function. therefore x should be vector.
    x = [x0, x1, x2, ..., xn].
	this function, _numerical_gradient, returns an array that consisted of each partial differential of fn/x.
    :param fn:
    :param x:
    :return:
    """
    # x = x.astype(np.float) # astype creates copy. 
    # print('_numerical_gradient x = ',x)
    gradient = np.zeros_like(x)  # np.zeros(shape=x.shape)
    h = 1e-4  # in order to prevent broadcast expansion, use for loop.

    for i in range(x.size):
        iterated_value = x[i]
        x[i] = iterated_value + h
        fh1 = fn(x)  # but somehow fn(x) has original x, not copied.
        x[i] = iterated_value - h
        fh2 = fn(x)  # as well.
        gradient[i] = (fh1 - fh2) / (2 * h)  # hence, differential becomes 0. Not astype float, initial inputs must be float. 
        x[i] = iterated_value
    return gradient


def numerical_gradient(fn, x):
    """x = [
        [x11 x12 x13, ...]
        [x21 x22 x23 ...]
        ...
    ]"""
    if x.ndim == 1:
        return _numerical_gradient(fn, x)
    else:
        gradi = np.zeros_like(x)
        for i, x_i in enumerate(x):
            # print(f'x_{i} = ', x_i)
            gradi[i] = _numerical_gradient(fn, x_i)
            # print(f'gradient {i}={gradi[i]}')
        return gradi