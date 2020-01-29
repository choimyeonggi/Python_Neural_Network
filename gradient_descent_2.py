"""
GRADIENT DESCENT 2.

refer page 137~.


"""

import inspect, os

print('CURRENT PATH FILES')
for i in os.listdir(os.curdir):
    print(i)

from gradient_descent_1 import numerical_gradient
import numpy as np
import activation_functions_module as afm
import loss_function_1 as lf1

class TwoLayerNet: # This code will be used after we defined.
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):

        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        p1 = afm.sigmoid(np.dot(x, W1) + b1)
        p2 = afm.softmax(np.dot(p1, W2) + b2)
        return p2

    def loss(self, x, t):
        return lf1.loss_cross_entropy(self.predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_Weight = lambda W : self.loss(x, t)
        gradients = dict()
        gradients['W1'] = gradient_descent_1.numerical_gradient(loss_Weight, self.params['W1'])
        gradients['b1'] = gradient_descent_1.numerical_gradient(loss_Weight, self.params['b1'])
        gradients['W2'] = gradient_descent_1.numerical_gradient(loss_Weight, self.params['W2'])
        gradients['b2'] = gradient_descent_1.numerical_gradient(loss_Weight, self.params['b2'])
        return gradients                
