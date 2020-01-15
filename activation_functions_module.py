"""
ACTIVATION FUNCTIONS
"""
import numpy as np

def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))


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

# added soft max : 20/1/15.
def softmax(a):
    c = np.max(a)
    a_prime = np.exp(a-c)
    return a_prime / np.sum(a_prime)
