"""
We are going to make simplest neural network system. for that, the basic idea is using linear algebra.

If a simple input like [x0, x1] is given, then we'd like to get 'proper' response as [y0, y1].

so the neural network we want to make can be treated as a function 'NW' that NW : [x0, x1] -> [y0, y1]. So the problem is what is the body of NW? by what we can 'properly adjust' inputs? Linear Algebra is the key for it.

Since [x0, x1] is 1*2 matrix, and the output [y0, y1] is also 1*2, We can guess that

the minimum structure of NW must be 2*
We are going to make simplest neural network system. for that, the basic idea is using linear algebra.

If a simple input like [x0, x1] is given, then we'd like to get 'proper' response as [y0, y1].

so the neural network we want to make can be treated as a function 'NW' that NW : [x0, x1] -> [y0, y1]. So the problem is what is the body of NW? by what we can 'properly adjust' inputs? Linear Algebra is the key for it.

Since [x0, x1] is 1*2 matrix, and the output [y0, y1] is also 1*2, We can guess that

the minimum structure of NW must be 2*2 matrix. (the size of matrices is neccessary information during dot product)

Anyway, we are going to construct the body of our neural network.
"""

import numpy as np


def init_network():
    # We need a dictionary that stores weights and biases for linear algebra operations . Since we just want a neural network body that works successfully anyway, weights and biases will be filled randomly.
    network = dict()
    network['W1'] = np.random.random(size=(2, 3)).round(
        2)  # size=(2,3) means "Row : 2, Col : 3". random number will be cut off at pow(10,-3).
    network['b1'] = np.random.random(3).round(
        2)  # Suppose that x (an input we give) is 1*2 matrix. Define @ as matrix product, the size of x @ W1 is 1*3. So, when we want to add a bias, its size must be 1*3. Hence the size was not set (2, 3).

    network['W2'] = np.random.random(size=(3, 2)).round(
        2)  # Second hidden layer. Remind that first hidden layer was x @ W1 + b1, size = 1*3, the row for W2 must be 3.
    network['b2'] = np.random.random(2).round(2)  # and the column of W2 is 2, bias(b2) should be 1*2.

    network['W3'] = np.random.random((2, 2)).round(2)  # Third hidden layer.
    network['b3'] = np.random.random(2).round(2)

    return network  # this is simple structure of neural network.


# We need a sigmoid function to adjust each hidden layer. (it decides characteristic of each neural network function)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# this function is always differentiable for all real numbers, fits for continuous data.

def softmax(x):
    xprime = x - np.max(x)
    return np.exp(xprime) / np.sum(np.exp(xprime))


def identity_function(x):
    return x


"""this function fits for classification(such as, is this image a cat? a dog?). the reason we adjust x into xprime(x - np.max(x)) is to avoid over/underflow, because exp is an exponential function. The equivalency can be proved easily by the properties of logarithm.

by the ESSENTIAL limitation of computer calculation, such as Overflow, underflow, floating points, it is important that you always have to consider data normalization in your mind.
"""


def forward_propagation(network, initial_input, sigmoid_function, return_sigmoid):
    """
    FORWARD PROPAGATION BASIC.


    :param network: basic dictionary that stores weights and biases. (At this point, We'll use init_network for it)
    :param initial_input: an initial input . at this point, 1*2 vector.
	:param sigmoid_function: a function which adjusts values from each hidden layer.
    :return: 'properly adjusted' response.
    """

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']  # can we generalize it?

    # Each matrix product means forward propagation. product possibility depends on 

    a1 = np.dot(initial_input, W1) + b1  # if initial_input is size of m*n, then W1 is n*l. hence b1 is size of m*l.
    # And then adjust a1 passing sigmoid_function
    z1 = sigmoid_function(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return return_sigmoid(y)  # passing sigmoid and return sigmoid can be distinguished.
