"""
ACTIVATION FUNCTIONS
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x, threshold=0.):
    if x.any() > threshold:
        y = x
    else:
        y = 0
    return y


def tanh(x):
    return np.tanh(x)


def arctan(x):
    return np.arctan(x)

def identity(x):
    return x
"""
# added soft max : 20/1/15.
def softmax(a):
    c = np.max(a)
    a_prime = np.exp(a-c)
    return a_prime / np.sum(a_prime)

added improved softmax : 20/1/16. referred from DeepLearning/ch03/azerates20191226c.py
the above one will be obsoleted.
"""
def softmax(x):
    """
    Suppose that input(x) is 1st-dimension vector, or matrix(dim=2).
    1) x  : [x1, x2, ... , xn]
    2) x  : [
            [x11, x12, ... , x1n],
            [x21, x22, ... , x2n],
            ...
            [xn1, xn2, ... , xnn]
             ]
    """
    dimension = x.ndim
    if dimension == 1:
        x = x - np.max(x) # rescale x, to prevent overflow.
        y = np.exp(x) / np.sum(np.exp(x))
    elif dimension == 2:
        # m = np.max(x, axis=1).reshape(len(x), 1)
        # # x = x - (x.T - np.max(x.T, axis=0)).T)
        # x = x - m
        # y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(len(x), 1)
        x = x.T - np.max(x.T, axis=0)
        y = (np.exp(x) / np.sum(np.exp(x), axis=0)).T
    return y
"""
    1. Axes of NumPy Arrays.
    
    axis=0 : row index (ascending order)
    axis=1 : column index

    See DeepLearning/ch03/azerates20191226b.py for detail.
"""
