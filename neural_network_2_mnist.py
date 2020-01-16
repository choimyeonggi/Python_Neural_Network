"""
MNIST

2020 01 15 JJH
"""

import os, sys
import inspect
import pickle
import numpy as np
from collections import OrderedDict
import activation_functions_module as afm

#Practice neural network with MNIST, Handwriting number image set.
print('Current Working Directory =', os.getcwd())  # C:\Users\Azerates\Documents\GitHub\STRZ_REPOSITORY_20191216

mnist_path = os.listdir(os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'dataset').replace('\\', '/'))
print('mnist_list_directory =',mnist_path)  #  ['mnist.pkl', 'mnist.py', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', '__init__.py', '__pycache__']
# we can load mnist.py directly.

print('relative mnist path =',os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'dataset').replace('\\', '/'))  # ../../../PycharmProjects/DeepLearning/dataset

# Now let's set dataset path to import mnist.py and others.
dataset_path = os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'dataset').replace('\\', '/')

sys.path.append(dataset_path)

sample_weight_path = os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'ch03', 'sample_weight.pkl').replace('\\', '/')

from mnist import load_mnist  # not dataset.mnist.

#pg 99.
from PIL import Image

#pg 100
import pickle

#although these processes are not PEP standard, linearly well congruent.
# to run this file, you must modify path and import configurations. download neccessary files here : https://github.com/WegraLee/deep-learning-from-scratch

def img_show(x):
    pil_img = Image.fromarray(np.uint8(x))
    pil_img.show()

class neural_network_1: # we will rename it.
    def __init__(self, initial_input, layers, output_size, activation_function=afm.sigmoid, output_function=afm.identity):

        self.x  = initial_input

        if IndexError:
            input_size = self.x.shape[0]
        else:
            input_size = self.x.shape[1]

        self.Weights = OrderedDict()
        self.Biases = OrderedDict()

        self.lst = []  # we need to use it again on product method.
        self.lst.append(input_size)
        
        for lyr in layers:
            self.lst.append(lyr)
        self.lst.append(output_size)
        
        for i in range(len(self.lst)-1):
            self.Weights['w'+str(i+1)] = np.random.randint(3, size=(self.lst[i], self.lst[i+1]))
            self.Biases['b'+str(i+1)] = np.random.randint(3, size=(1, self.lst[i+1]))
            
        print('Weight Dictionary =', self.Weights)
        print('Bias Dictionary =', self.Biases)
            
        self.activator = activation_function
        self.output_controller = output_function

    def product(self):
        product_iterand = self.x
        product_list = []
        
        for k in range(len(self.lst)-1):
            if self.lst[k]:
                output_itr = self.activator(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            else:
                output_itr = self.output_controller(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            print(f'\n iterand : {product_iterand},\n Weight : \n {self.Weights["w"+str(k+1)]}, \n Bias : {self.Biases["b"+str(k+1)]}')
            print(f'\n iterand : {product_iterand.shape}, Weight : {self.Weights["w"+str(k+1)].shape}, Bias : {self.Biases["b"+str(k+1)].shape}')
            #print(f'\n iterand : {product_iterand, product_iterand.shape},\n Weight : \n {self.Weights["w"+str(k+1)], self.Weights["w"+str(k+1)].shape}, \n Bias : {self.Biases["b"+str(k+1)], self.Biases["b"+str(k+1)].shape}')
            product_iterand = output_itr
            product_list.append(output_itr)
            print(f'output =' ,output_itr)
            print('------------------------------------------------------------')
        return output_itr, product_list
    """
This structure, has a limitation. That, you cannot set weights and biases as you want, but random numbers.
    """

#refer pg 100.
def init_network():
    with open(sample_weight_path, mode='rb') as fi:
        network = pickle.load(fi)
        print(network.keys())
    return network

def predict(ntw, x):
    W1, W2, W3 = ntw['W1'],ntw['W2'],ntw['W3']
    b1, b2, b3 = ntw['b1'],ntw['b2'],ntw['b3']

    a1 = np.dot(x, W1) + b1
    z1 = afm.sigmoid(a1)
    
    a2 = np.dot(z1, W2) + b2
    z2 = afm.sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    z3 = afm.softmax(a3)
    
    return z3


def accuracy_1(predicted, answers, network, f_prop):
    """
    predicted = data we want to predict. Passing neural network one by one.
    answers = which we want to compare to predicted.
    """
    if predicted.shape[0] == len(answers):
        cnt = 0
        for i in range(len(predicted)):
            comparison = np.argmax(f_prop(network, predicted[i]))
            if comparison == answers[i]:
                cnt += 1
    else:
        raise ValueError('Two parameters must have same length.')
    return cnt / len(predicted)

if __name__ == '__main__':

    # Now let us see source code of load_mnist.

    #print('\n source code of load_mnist \n', inspect.getsource(load_mnist))
    # refer "밑바닥부터 시작하는 딥러닝" published by O'REILLY, 한빛미디어.  pg 97.

    (x_train , y_train), (x_test, y_test) = load_mnist(normalize=True)

    print('x_train.shape =', x_train.shape, '\n', 'y_train.shape =', y_train.shape, '\n', 'x_test.shape =', x_test.shape, '\n', 'y_test.shape =', y_test.shape)
    """
     x_train.shape = (60000, 784) 
     y_train.shape = (60000,) 
     x_test.shape = (10000, 784) 
     y_test.shape = (10000,)
    """

    #transform data into specific images.
    #random_image_number = np.random.randint(x_train.shape[0])  # take any random number within 1 ~ 60000.
    #img_example= x_train[random_image_number].reshape(28, 28)
    #img_show(img_example)

    """
    Now let us improve our function, neural_network_1.single_layer_3

    class single_layer_4:
    def __init__(self, initial_input, layers, output_size, activation_function=afm.sigmoid, output_function=afm.identity):

        self.x = initial_input

        if IndexError:
            input_size = self.x.shape[0]
        else:
            input_size = self.x.shape[1]

        self.Weights = OrderedDict()
        self.Biases = OrderedDict()

        self.lst = []  # we need to use it again on product method.
        self.lst.append(input_size)
        
        for lyr in layers:
            self.lst.append(lyr)
        self.lst.append(output_size)
        
        for i in range(len(self.lst)-1):
            self.Weights['w'+str(i+1)] = np.random.randint(10, size=(self.lst[i], self.lst[i+1]))
            self.Biases['b'+str(i+1)] = np.random.randint(10, size=(1, self.lst[i+1]))
        self.activator = activation_function
        self.output_controller = output_function

    def product(self):
        product_iterand = self.x
        product_list = []
        
        for k in range(len(self.lst)-1):
            if self.lst[k]:
                output_itr = self.activator(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            else:
                output_itr = self.output_controller(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            product_iterand = output_itr
            product_list.append(output_itr)
        return output_itr, product_list
>> added output function.
At least, it seems that our function works well, no calculation error.
    """
    #sl_input = np.array([1, 1])
    #sl = neural_network_1(sl_input, layers=[2, 3, 4, 5], output_size=6, activation_function=afm.sigmoid)
    #sl.product()
    #print('\n')

    sample_weight = init_network()

    print(x_test[0].shape) # 1 784
    
    print(sample_weight['W1'].shape) #784 50
    print(sample_weight['W2'].shape) # 50 100
    print(sample_weight['W3'].shape) # 100 10

    print(sample_weight['b1'].shape) # 1 50
    print(sample_weight['b2'].shape) # 1 100
    print(sample_weight['b3'].shape) # 1 10

   # accuracy from given sample weight.
    accuracy_count= 0
    for i in range(len(x_test)):
        x = predict(sample_weight, x_test[i])
        p = np.argmax(x)
        if p == y_test[i]:
            accuracy_count += 1
    print('accuracy =', float(accuracy_count / len(x_test)))  # accuracy = 0.9352

    # We can make it more generally, by function form. refer accuracy_1. the function below is referred from DeepLearning/ch03/azerates20191224d.py

    accuracy_f = accuracy_1(predicted=x_test, answers=y_test, network=sample_weight, f_prop=predict)
    print('accuracy by accuracy_1 function =', accuracy_f)  # accuracy by accuracy_1 function = 0.9352
    
