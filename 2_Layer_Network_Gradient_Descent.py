"""

2-Layer Neural Network Basic

We are going to look into the basic structure of Neural network that has 2 layers. By this, we can expand layers as many as we want, theoritically. 
"""

import numpy as np
from dataset.mnist import load_mnist


class TwoLayerNetwork():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        np.random.seed(1231)
        """
		Input 784, 1st layer has 32 neurons, output layer has 10.
        
        1*784 -> 784 * 32 -> 32*10
        
        1st weighted matrix : 784*32
        1st bias : 1*32
        
        output matrix : 32*10
        output bias : 1*10
        """
        self.parametres = dict()
        self.parametres['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.parametres['b1'] = np.zeros(hidden_size)
        self.parametres['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.parametres['b2'] = np.zeros(output_size)

    # Initialised weights with random numbers, biases 0. Can we generalise this ? maybe can, but we don't need it now.
	
    def sigmoid(self, x):
        return np.array(1 / (1 + np.exp(-x)))

    def softmax(self, x):
        """softmax = exp(x_k) / sum(exp(x_i))"""
        dimension = x.ndim
        if dimension == 1:
            x = x - np.max(x)  # To prevent overflow, rescale by maximum.
            y = np.exp(x) / np.sum(np.exp(x))
        elif dimension == 2:	
            x = x.T - np.max(x.T, axis=0)
            y = (np.exp(x) / np.sum(np.exp(x), axis=0)).T
        return y  # we can simplise this code with np.nditer, but not now.

    def predict(self, datum):
        """
		put input data(array) into neural network structure, passing matrix product, return output(array).

        z1 = sigmoid(data.dot(w1) + b1)
        z2 = softmax(z1.dot(w2) + b2)

        :return:
        """
        self.z1 = datum.dot(self.parametres['W1']) + self.parametres['b1']
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.z1.dot(self.parametres['W2'] + self.parametres['b2'])
        self.a2 = self.softmax(self.z2)   #output.
        return self.a2

    def accuracy(self, datum, y_true):
        """  """
        # return np.mean(self.predict(datum) == y_true)
        pass
        y_pred = self.predict(datum)
        predictions = np.argmax(y_pred, axis=1)  # take maximum by ROW(axis=1).
        trues = np.argmax(y_true, axis=1)
        print('predictions = ', predictions)
        print('trues = ', trues)
        acr = np.mean(predictions == trues)
        return acr

    def cross_entropy(self, datum, y_true, delta=1e-7):
        y_pred = self.predict(datum)
        entropy = -np.sum(y_true * np.log(y_pred + delta))
        if y_pred.ndim == 1:
            ce = entropy
        elif y_pred.ndim == 2:
            ce = entropy / len(y_pred)
        return ce

        """
        in other words
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape((1, y_pred.size))
            y_true = y_true.reshape((1, y_true.size))  # suppose that one_hot_encoding=True
        true_values = np.argmax(y_true, axis=1)  # axis=1 -> row.
        n = y_pred.shape[0]  # number of row from y_pred.
        rows = np.arange(n)  # [0, 1, 2, 3, ... , n]
        logp = np.log(y_pred[rows, true_values])  # y_pred[ [0, 1, 2, ... ], [3, 3, 9, ... ] ] <-> [y_pred[0, 3], y_pred[1, 3], y_pred[3, 9], ... ]
        cross_entropy = -np.sum(logp)/n
        return cross_entropy
        """

    def loss(self, datum, y_true):
        entropy = self.centropy(datum, y_true)
        return entropy

    def numerical_gradient(self, fn, x, delta=1e-4):
        """USE np.nditer"""
        gradient = np.zeros_like(x)
        # Create an iterator about x.
        with np.nditer(x, flags=['c_index', 'multi_index'], op_flags=['readwrite']) as itr:
            while not itr.finished:
                index = itr.multi_index
                value = itr[0]  # store original data into temporal variable
                itr[0] = value + delta  # increases it with delta.
                fh1 = fn(x)  # this is then forward differential.
                itr[0] = value - delta  # decreases it with delta.
                fh2 = fn(x)  # this is then backward differential.
                gradient[index] = (fh1 - fh2) / (2 * delta)
                itr[0] = value
                itr.iternext()
        return gradient

    def gradients(self, datum, y_true):
        loss_fn = lambda W: self.loss(datum, y_true)
        gradientes = dict()
        # gradientes['W1'] = self.numerical_gradient(loss_fn, self.parametres['W1'])
        # gradientes['b1'] = self.numerical_gradient(loss_fn, self.parametres['b1'])
        # gradientes['W2'] = self.numerical_gradient(loss_fn, self.parametres['W2'])
        # gradientes['b1'] = self.numerical_gradient(loss_fn, self.parametres['b2'])
        for i in self.parametres.keys():
            gradientes[i] = self.numerical_gradient(loss_fn, self.parametres[i])
        return gradientes


if __name__ == '__main__':
    # print(ntw2.predict(np.array(np.random.randint(1, 784))))

    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)
    print(X_train.shape)

    ntw2 = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
    print('W1 :', ntw2.parametres['W1'].shape)
    print('b1 :', ntw2.parametres['b1'].shape)
    print('W2 :', ntw2.parametres['W2'].shape)
    print('b2 :', ntw2.parametres['b2'].shape)
    # print(ntw2.predict(X_train[0]))
    # print(ntw2.predict(X_train[:5]))
    array_5 = ntw2.predict(X_train[:5])
    print(array_5)
    print(np.argmax(array_5, axis=1))
    print(np.argmax(array_5, axis=0))
    print('accuracy =', ntw2.accuracy(X_train[:500], y_train[:500]))
    print(ntw2.loss(X_train, y_train))
    print(ntw2.loss(X_train[:5], y_train[:5]))
    print(ntw2.loss(X_train[:100], y_train[:100]))

    print(ntw2.parametres.keys())
    # print(ntw2.gradients(X_train[:100], y_train[:100]))  # get gradient to update weights and biases.

    gradients = ntw2.gradients(X_train[:100], y_train[:100])
    for ky in gradients:
        print(ky, np.sum(gradients[ky]))

    learning_rate = 0.1
    for k in gradients:
        ntw2.parametres[k] -= learning_rate * gradients[k]
    print(ntw2.parametres)  # Since gradient descent handles big size matrices, takes comparatively long time to iterate 1 epoch.

    # epoch = 1000
    # for ep in range(epoch):
    #     for ix in range(600):
    #         gradiens = ntw2.gradients(X_train[ix * 100:(ix + 1) * 100], y_train[ix * 100:(ix + 1) * 100])
    #         for key in gradiens:
    #             ntw2.parametres[key] -= learning_rate * gradients[key]
