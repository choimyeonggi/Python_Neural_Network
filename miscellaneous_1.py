"""
MISCELLANEOUS_1

2020 01 15 JJH
"""
import os
import inspect
import numpy as np
import pandas as pd
from collections import OrderedDict
import activation_functions_module as afm
import neural_network_1 as nn_1
import matplotlib as plt

# Last time we made a single layer neural network. Refer neural_network_1.py for further comments.

class single_layer_3:
    def __init__(self, initial_input, layers, output_size, activation_function=afm.sigmoid):

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

    def product(self):
        product_iterand = self.x
        product_list = []
        
        for k in range(len(self.lst)-1):
            output_itr = self.activator(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            product_iterand = output_itr
            product_list.append(output_itr)
        return output_itr, product_list


if __name__ == '__main__':
    np.random.seed(115)

    '''
before that, let us take a look at inspect module. this can be very useful  as we are going to write codes on basic IDLE. 
'''
    #print(inspect.getmembers(afm, inspect.isfunction))
    """
[('arctan', <function arctan at 0x0000017C957F28B8>),
('identity', <function identity at 0x0000017C957F2948>),
('relu', <function relu at 0x0000017C86DCA0D8>),
('sigmoid', <function sigmoid at 0x0000017C86DCA318>),
('tanh', <function tanh at 0x0000017C957F2828>)]

> show all available functions from the targeted module. For that, you must previously import it.


"""
    #print(inspect.getmembers(nn_1, inspect.isclass))
    """
[('OrderedDict', <class 'collections.OrderedDict'>),
('single_layer_1', <class 'neural_network_1.single_layer_1'>),
('single_layer_2', <class 'neural_network_1.single_layer_2'>),
('single_layer_3', <class 'neural_network_1.single_layer_3'>)]

> show all available classes from the targeted module. We have imported neural_network_1 as nn_1. 
"""
    #print(inspect.getdoc(nn_1.single_layer_3))
    """
    e.g.
initial_input = np.array([5, 7, 8 ,3, 4])
layers = [3, 5]
output_size = 3
> see document of targeted function.
    """
    #print(inspect.getsource(nn_1.single_layer_1))
    """
class single_layer_1:
    def __init__(self,
                 initial_input=np.array([1, 2]),
                 layer_1=np.random.randint(10, size=(2, 3)),
                 bias_1=np.random.randint(10, size=(1, 3)),
                 activation_function=afm.sigmoid):
        self.x = initial_input
        self.w = layer_1
        self.b = bias_1
        self.activator = activation_function  # now we can call these variables as much as we want.
        #print('x =', self.x, self.x.shape[0])
        #print('w =', self.w, self.w.shape)
        #print('b =', self.b, self.b.shape)
    
    def product(self):
        # output = self.x.dot(self.w) + self.b
        output = self.activator(self.x.dot(self.w) + self.b)
        return output
> show sourcecode of targeted function, or class, any other object.
    """
    print(inspect.signature(nn_1.single_layer_3))
    """
(initial_input, layers, output_size, activation_function=<function sigmoid at 0x00000252F432A318>)
> show arguments of targeted object.
    """
    #print(os.getcwd()) # C:\Users\Azerates\Documents\GitHub\STRZ_REPOSITORY_20191216
    #os.system('notepad')
    #files = os.listdir(os.curdir)  #files and directories
    #print(files)
    """
['.git',
'.gitattributes',
'.github',
'2_Layer_Network_Gradient_Descent.py',
'activation_functions_module.py',
'Apply_Gradient_Descent_To_Neural_Network.py',
'Apply_Gradient_Descent_To_Neural_Network_module.py',
'azerates20191122a.py',
'basic_neuron_practice.py',
'cointoss.py',
'forward_propagation_001.py',
'forward_propagation_002.py',
'neural_network_1.py',
'neural_network_2.py',
'README.md',
'__pycache__']

> source link : https://stackoverflow.com/questions/11968976/list-files-only-in-the-current-directory
we can now import files by this not using explorers
    """
    #for i in files:
      #  print(i)
    """
Same result, but aligned.

.git
.gitattributes
.github
2_Layer_Network_Gradient_Descent.py
activation_functions_module.py
Apply_Gradient_Descent_To_Neural_Network.py
Apply_Gradient_Descent_To_Neural_Network_module.py
azerates20191122a.py
basic_neuron_practice.py
cointoss.py
forward_propagation_001.py
forward_propagation_002.py
neural_network_1.py
neural_network_2.py
README.md
__pycache__
    """
    # print(inspect.getsource(nn_1)) # shows all source code. not recommended.
    # print(inspect.getsource(np))
    print('version 3 test')
    sl_input = np.array([0.1,0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
    sl = single_layer_3(
                        initial_input=sl_input,
                        layers=[3, 4, 3, 2, 6, 4, 10, 4, 3, 5, 2],
                        output_size=10,
                        activation_function=afm.softmax)
    x = sl.product()[0]
    print( (x - np.mean(x)) / np.std(x) )  # z-centralisation
    print( (x - np.min(x)) / (np.max(x) - np.min(x))) # min-max normalization.
    # it seems, we may not need to add z-centralisation or min-max normalization
    print(x)
    print(np.sum(afm.softmax(sl_input))) # = 1.0
    print('\n')

    """How can we access files, which are not on this directory?
"""
    #print(inspect.getsource(os.path))
    print(os.path.relpath(
        'C:/Users/Azerates/PycharmProjects/ITWILL/SCRATCH_LECT_11', os.getcwd()
    )) #..\..\..\PycharmProjects\ITWILL\SCRATCH_LECT_11

    files = os.listdir(os.path.join('..', '..', '..','PycharmProjects', 'ITWILL', 'SCRATCH_LECT_11'))
    for i in files:
        print(i)
        """
azerates20191211a.py
breast-cancer-wisconsin-data.csv
iris.csv
KNN Classifier Reinvention azerates20191210c.py
Scikit Learn KNN azerates20191210a_knn.py
Scikit Learn Wisconsin Breast Cancer azerates20191210b_knn.py
__init__.py
        """

    iris_path = os.path.join('..', '..', '..','PycharmProjects', 'ITWILL', 'SCRATCH_LECT_11', 'iris.csv').replace('\\', '/')
    print('iris_path =',iris_path) # iris_path = ../../../PycharmProjects/ITWILL/SCRATCH_LECT_11/iris.csv

    colnames = [
        'sepal-length',
        'sepal-width',
        'petal-length',
        'petal-width',
        'Class'
    ]

    iris_path_data = pd.read_csv(iris_path, header=None, encoding='UTF-8', names=colnames)
    print(type(iris_path_data))
# refer from https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    for index, row in iris_path_data.iterrows():
        print(row[colnames[0]], row[colnames[1]], row[colnames[2]], row[colnames[3]], row[colnames[4]])
    """
result:
5.1 3.5 1.4 0.2 Iris-setosa
4.9 3.0 1.4 0.2 Iris-setosa
4.7 3.2 1.3 0.2 Iris-setosa
4.6 3.1 1.5 0.2 Iris-setosa
5.0 3.6 1.4 0.2 Iris-setosa
5.4 3.9 1.7 0.4 Iris-setosa
4.6 3.4 1.4 0.3 Iris-setosa
5.0 3.4 1.5 0.2 Iris-setosa
4.4 2.9 1.4 0.2 Iris-setosa
4.9 3.1 1.5 0.1 Iris-setosa
5.4 3.7 1.5 0.2 Iris-setosa
4.8 3.4 1.6 0.2 Iris-setosa
4.8 3.0 1.4 0.1 Iris-setosa
4.3 3.0 1.1 0.1 Iris-setosa
5.8 4.0 1.2 0.2 Iris-setosa
5.7 4.4 1.5 0.4 Iris-setosa
5.4 3.9 1.3 0.4 Iris-setosa
5.1 3.5 1.4 0.3 Iris-setosa
5.7 3.8 1.7 0.3 Iris-setosa
5.1 3.8 1.5 0.3 Iris-setosa
5.4 3.4 1.7 0.2 Iris-setosa
5.1 3.7 1.5 0.4 Iris-setosa
4.6 3.6 1.0 0.2 Iris-setosa
5.1 3.3 1.7 0.5 Iris-setosa
4.8 3.4 1.9 0.2 Iris-setosa
5.0 3.0 1.6 0.2 Iris-setosa
5.0 3.4 1.6 0.4 Iris-setosa
5.2 3.5 1.5 0.2 Iris-setosa
5.2 3.4 1.4 0.2 Iris-setosa
4.7 3.2 1.6 0.2 Iris-setosa
4.8 3.1 1.6 0.2 Iris-setosa
5.4 3.4 1.5 0.4 Iris-setosa
5.2 4.1 1.5 0.1 Iris-setosa
5.5 4.2 1.4 0.2 Iris-setosa
4.9 3.1 1.5 0.1 Iris-setosa
5.0 3.2 1.2 0.2 Iris-setosa
5.5 3.5 1.3 0.2 Iris-setosa
4.9 3.1 1.5 0.1 Iris-setosa
4.4 3.0 1.3 0.2 Iris-setosa
5.1 3.4 1.5 0.2 Iris-setosa
5.0 3.5 1.3 0.3 Iris-setosa
4.5 2.3 1.3 0.3 Iris-setosa
4.4 3.2 1.3 0.2 Iris-setosa
5.0 3.5 1.6 0.6 Iris-setosa
5.1 3.8 1.9 0.4 Iris-setosa
4.8 3.0 1.4 0.3 Iris-setosa
5.1 3.8 1.6 0.2 Iris-setosa
4.6 3.2 1.4 0.2 Iris-setosa
5.3 3.7 1.5 0.2 Iris-setosa
5.0 3.3 1.4 0.2 Iris-setosa
7.0 3.2 4.7 1.4 Iris-versicolor
6.4 3.2 4.5 1.5 Iris-versicolor
6.9 3.1 4.9 1.5 Iris-versicolor
5.5 2.3 4.0 1.3 Iris-versicolor
6.5 2.8 4.6 1.5 Iris-versicolor
5.7 2.8 4.5 1.3 Iris-versicolor
6.3 3.3 4.7 1.6 Iris-versicolor
4.9 2.4 3.3 1.0 Iris-versicolor
6.6 2.9 4.6 1.3 Iris-versicolor
5.2 2.7 3.9 1.4 Iris-versicolor
5.0 2.0 3.5 1.0 Iris-versicolor
5.9 3.0 4.2 1.5 Iris-versicolor
6.0 2.2 4.0 1.0 Iris-versicolor
6.1 2.9 4.7 1.4 Iris-versicolor
5.6 2.9 3.6 1.3 Iris-versicolor
6.7 3.1 4.4 1.4 Iris-versicolor
5.6 3.0 4.5 1.5 Iris-versicolor
5.8 2.7 4.1 1.0 Iris-versicolor
6.2 2.2 4.5 1.5 Iris-versicolor
5.6 2.5 3.9 1.1 Iris-versicolor
5.9 3.2 4.8 1.8 Iris-versicolor
6.1 2.8 4.0 1.3 Iris-versicolor
6.3 2.5 4.9 1.5 Iris-versicolor
6.1 2.8 4.7 1.2 Iris-versicolor
6.4 2.9 4.3 1.3 Iris-versicolor
6.6 3.0 4.4 1.4 Iris-versicolor
6.8 2.8 4.8 1.4 Iris-versicolor
6.7 3.0 5.0 1.7 Iris-versicolor
6.0 2.9 4.5 1.5 Iris-versicolor
5.7 2.6 3.5 1.0 Iris-versicolor
5.5 2.4 3.8 1.1 Iris-versicolor
5.5 2.4 3.7 1.0 Iris-versicolor
5.8 2.7 3.9 1.2 Iris-versicolor
6.0 2.7 5.1 1.6 Iris-versicolor
5.4 3.0 4.5 1.5 Iris-versicolor
6.0 3.4 4.5 1.6 Iris-versicolor
6.7 3.1 4.7 1.5 Iris-versicolor
6.3 2.3 4.4 1.3 Iris-versicolor
5.6 3.0 4.1 1.3 Iris-versicolor
5.5 2.5 4.0 1.3 Iris-versicolor
5.5 2.6 4.4 1.2 Iris-versicolor
6.1 3.0 4.6 1.4 Iris-versicolor
5.8 2.6 4.0 1.2 Iris-versicolor
5.0 2.3 3.3 1.0 Iris-versicolor
5.6 2.7 4.2 1.3 Iris-versicolor
5.7 3.0 4.2 1.2 Iris-versicolor
5.7 2.9 4.2 1.3 Iris-versicolor
6.2 2.9 4.3 1.3 Iris-versicolor
5.1 2.5 3.0 1.1 Iris-versicolor
5.7 2.8 4.1 1.3 Iris-versicolor
6.3 3.3 6.0 2.5 Iris-virginica
5.8 2.7 5.1 1.9 Iris-virginica
7.1 3.0 5.9 2.1 Iris-virginica
6.3 2.9 5.6 1.8 Iris-virginica
6.5 3.0 5.8 2.2 Iris-virginica
7.6 3.0 6.6 2.1 Iris-virginica
4.9 2.5 4.5 1.7 Iris-virginica
7.3 2.9 6.3 1.8 Iris-virginica
6.7 2.5 5.8 1.8 Iris-virginica
7.2 3.6 6.1 2.5 Iris-virginica
6.5 3.2 5.1 2.0 Iris-virginica
6.4 2.7 5.3 1.9 Iris-virginica
6.8 3.0 5.5 2.1 Iris-virginica
5.7 2.5 5.0 2.0 Iris-virginica
5.8 2.8 5.1 2.4 Iris-virginica
6.4 3.2 5.3 2.3 Iris-virginica
6.5 3.0 5.5 1.8 Iris-virginica
7.7 3.8 6.7 2.2 Iris-virginica
7.7 2.6 6.9 2.3 Iris-virginica
6.0 2.2 5.0 1.5 Iris-virginica
6.9 3.2 5.7 2.3 Iris-virginica
5.6 2.8 4.9 2.0 Iris-virginica
7.7 2.8 6.7 2.0 Iris-virginica
6.3 2.7 4.9 1.8 Iris-virginica
6.7 3.3 5.7 2.1 Iris-virginica
7.2 3.2 6.0 1.8 Iris-virginica
6.2 2.8 4.8 1.8 Iris-virginica
6.1 3.0 4.9 1.8 Iris-virginica
6.4 2.8 5.6 2.1 Iris-virginica
7.2 3.0 5.8 1.6 Iris-virginica
7.4 2.8 6.1 1.9 Iris-virginica
7.9 3.8 6.4 2.0 Iris-virginica
6.4 2.8 5.6 2.2 Iris-virginica
6.3 2.8 5.1 1.5 Iris-virginica
6.1 2.6 5.6 1.4 Iris-virginica
7.7 3.0 6.1 2.3 Iris-virginica
6.3 3.4 5.6 2.4 Iris-virginica
6.4 3.1 5.5 1.8 Iris-virginica
6.0 3.0 4.8 1.8 Iris-virginica
6.9 3.1 5.4 2.1 Iris-virginica
6.7 3.1 5.6 2.4 Iris-virginica
6.9 3.1 5.1 2.3 Iris-virginica
5.8 2.7 5.1 1.9 Iris-virginica
6.8 3.2 5.9 2.3 Iris-virginica
6.7 3.3 5.7 2.5 Iris-virginica
6.7 3.0 5.2 2.3 Iris-virginica
6.3 2.5 5.0 1.9 Iris-virginica
6.5 3.0 5.2 2.0 Iris-virginica
6.2 3.4 5.4 2.3 Iris-virginica
5.9 3.0 5.1 1.8 Iris-virginica
    """
    
