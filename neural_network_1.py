"""
NEURAL NETWORK_1

2020 01 14 JJH
"""
import sys
import numpy as np
from collections import OrderedDict
import activation_functions_module as afm


# Version 1

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

    # can we generalise, that hidden layers change by the shape of input?


"""
1. If x is single row, then there is no index 1. So we can switch.

define 2 variables : input_size, number_of_neurons.

if self.x.shape[1] == False:  # if self.x.shape[1] is out of bound
    input_size = self.x.shape[0]
else:
    input_size = self.x.shape[1]

-> this is the point.

"""


# Version 2
class single_layer_2:
    def __init__(self,
                 initial_input,
                 neurons,
                 activation_function=afm.sigmoid):

        self.x = initial_input

        if IndexError:
            input_size = self.x.shape[0]
        else:
            input_size = self.x.shape[1]

        self.w = np.random.randint(10, size=(input_size, neurons))
        self.b = np.random.randint(10, size=(1, neurons))
        self.activator = activation_function

    def product(self):
        output = self.activator(self.x.dot(self.w) + self.b)
        return output

"""
We don't need to specify weights and biases of neural network. Because they will be automatically rescaled by machine learning techniques like back propagation.

Can we set hidden layers as many as we want?

if hidden_layers = 1, then it is totally same above. (the parametre must be natural number.)

if hidden_layers is more than 2,

self.w1 = np. random.randint(10, size=(input_size, neurons_1))
self.w2 = np.random.randint(10, size=(neurons_1, neurons2))
self.w3 = np.random.randint(10, size=(neurons_2, neurons3))
...

We may need a dictionary variable for this.

neurons = [3, 5]

self.Weights = OrderedDict()  -> # from collections import OrderedDict

# self.Weights['w1']=np.random.randint(10, size=(input_size, neurons[0]))  -> deprecated

for i in range(hidden_layers):  # if hidden_layers=2, 0, 1. -> deprecated, now len(neurons).
    self.Weights['w'+str(i+1)] = np.random.randint(10, size=(neurons[i], neurons[i+1]))  # self.Weights['w1'] = np.random.randint(10, size=(neurons[0], neurons[1])) -> input size must be placed to neurons[0].
    
hence

    lst = []
    lst.append(input_size)
    for nr in neurons:
        lst.append(i)  # since append may take single element at once. and len(lst) = len(neurons)+1.

    for i in range(len(lst)):
        self.Weights['w'+str(i+1)] = np.random.randint(10, size=(neurons[i], neurons[i+1]))

    
    and we need to make biases either. this is relatively simple.
    make an orderd dictionary for biases.

self.Biases = OrderedDict()

for j in range(len(lst)):
    self.Biases['b'+str(j+1)] = np.random.randint(10, size=(1, neurons[i+1]))  # self.b1 = np.random.randint(10, size=(1, 3))

    
neurons -> layers.
"""

# Version 3 (final)

class single_layer_3:

    """
    e.g.
    initial_input = np.array([5, 7, 8 ,3, 4])
    layers = [3, 5]
    output_size = 3
    
    """
    def __init__(self,
                 initial_input,
                 layers,
                 output_size,
                 activation_function=afm.sigmoid):

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
        
        print('hidden layers structure :',self.lst, len(self.lst))
        
        for i in range(len(self.lst)-1):
            
            #print('i=', i)  # suppose that len(lst)=3 when i=2, there is no layers[3].
            
            self.Weights['w'+str(i+1)] = np.random.randint(10, size=(self.lst[i], self.lst[i+1]))
            self.Biases['b'+str(i+1)] = np.random.randint(10, size=(1, self.lst[i+1]))   #trouble occurs at i=2. ah, we should take lst, not layers!
            
        print('Weight Dictionary =', self.Weights)
        print('Bias Dictionary =', self.Biases)
        
        self.activator = activation_function

    def product(self):
        """  Weights and biases must be producted step by step.
the "output" is consisted of composition of function. this method can be called in other words 'forward propagation'.

optimizing
output = self.activator(self.x.dot(self.w) + self.b),

1>output_1 = self.activator(self.x.dot(self.Weights['w1'])) + self.Biases['b1']
2>output_2 = self.activator(output_1.dot(self.Weights['w2'])) + self.Biases['b2']
3>output_3 = self.activator(output_2.dot(self.Weights['w3'])) + self.Biases['b3']


can we iterate it?
product_iterand = self.x
product_list =[]
for k in range(len(lst)-1):
    output_itr = self.activator(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
    product_iterand = output_itr
    product_list.append(output_itr)
    '''first iteration :  output_itr = self.activator(self.x.dot(self.Weights['w1']) + self.Biases['b1']
    product_iterand : self.x -> output_itr_1(with w1,b1)
    product_list : [output_itr_1]'''
    '''second iteration :  output_itr = self.activator(output_itr_1(self.Weights['w2']) + self.Biases['b2']
    product_iterand : output_itr_1 -> output_itr_2
    product_list : [output_itr_1, output_itr_2]'''
    
    

the function returns its final output and product list. so if you'd like to extract output only,type sl.product()[0].
"""
        product_iterand = self.x
        product_list = []
        for k in range(len(self.lst)-1):
            output_itr = self.activator(product_iterand.dot(self.Weights['w'+str(k+1)]) + self.Biases['b'+str(k+1)])
            product_iterand = output_itr
            product_list.append(output_itr)
            print(output_itr)
            
        return output_itr, product_list
            

"""
self.w1 = np. random.randint(10, size=(layers[0], layers[1]))
self.w2 = np.random.randint(10, size=(layers[1],layers[2]))
self.w3 = np.random.randint(10, size=(layers[2],layers[3])) <<< error point. << not layers, but self.lst.

-> we need to define OUTPUT SIZE.

suppose that len(layers)=2

then len(lst) = 4. hence range(len(lst)) => 0, 1, 2, 3. thus range must be range(len(lst)-1).

len(lst)=3. index : 0, 1, 2.


Can we differentiate activation function layer by layer?
"""
        


if __name__ == '__main__':
    np.random.seed(114)

    """
    to run this module normally, you need to import activation_functions_module either.

    Last time, We made a simple neuron that accepts a single value from input. What if we make a multi-layered networks with it?


    Suppose that we'd like to input 2 values at once, i = [1, 2].



    """

    i = np.array([1, 2])

    print(i, i.shape)  # [1, 2]

    """ Now let us make a weight matrix. We'll suppose that there are 3 neurons to pass through. 

                neuron1
    x1  
                neuron2
    x2
                neuron3

    To connect x1 and neuron1, it can be presented by x1 * w1 + b1.
    neuron2, x1 * w2 + b2
    neuron3, x1 * w3 + b3.

    as same as x2, 
    neuron1, x2 * w4 + b4
    neuron2, x2 * w5 + b5
    neuron3, x2 * w6 + b6.

    -> neuron1 : x1*w1 + x2*w4 + b1
        neuron2 : x1*w2 + x2*w5 + b2
        neuron3 : x1*w3 + x2*w6 + b3
                                                        -> generalised : neuronN : x1*wN + x2*w(N+3) + bN.
                                                        -> Activation = Input * Weight + Bias

    hence, the neuron layer is 
    ( w1, w2, w3
    w4, w5, w6)

    so is bias layer.

    (b1, b2, b3
    b4, b5, b6).

    from this we can know one important character : the number of neurons is the column for neural layer.

    if size(input)=(1, 2) and size(neural_layer1)=(2,3) then size(input*neural_layer_1)=(1,3). therefore size(bias)=(1,3).

    now, let's make neural_layer1.
    """
    neural_layer_1 = np.random.randint(10, size=(2, 3))  # [[3 6 2] [2 8 9]] (2, 3)
    print(neural_layer_1, neural_layer_1.shape)

    print('input * weight =', i.dot(neural_layer_1))  # [ 7, 22, 20]

    # let us make bias matrix either.

    bias_layer_1 = np.random.randint(10, size=(1, 3))
    print('bias layer 1 =', bias_layer_1, bias_layer_1.shape)  # [3 8 4]

    # hence, the activation is
    activation_1 = i.dot(neural_layer_1) + bias_layer_1  # activation = input * weight + bias.
    print('activation 1 =', activation_1)  # [10, 30, 24]

    print('passing sigmoid :', afm.sigmoid(activation_1))
    print('passing relu :', afm.relu(activation_1))
    print('passing tanh :', afm.tanh(activation_1))
    print('passing arctan :', afm.arctan(activation_1))
    '''
passing sigmoid : [[0.9999546 1.        1.       ]]
passing relu : [[10 30 24]]
passing tanh : [[1. 1. 1.]]
passing arctan : [[1.47112767 1.53747533 1.52915375]]
'''
    # print(sys.path)

    """

    How can we make it as function, or class? It seems like we can make it as class.

    """

    # version1
    
    sl = single_layer_1(activation_function=afm.identity)
    print('version 1 single layer product =', sl.product())
#version 1 single layer product = [[ 6 17 28]]
    # version 2
    
    i = np.array([1, 2, 3, 4, 5, 6])
    #i = np.array([1])
    sl = single_layer_2(initial_input=i, neurons=5, activation_function=afm.identity)
    print('version 2 single layer product =', sl.product())
#version 2 single layer product = [[103  58 117 107  93]]
    print('version 3 test 1')
    sl_input = np.array([1,2])
    sl = single_layer_3(sl_input, [3, 4], output_size=3)
    sl.product()
    print('\n')

    print('version 3 test 2')
    sl_input = np.array([1,2, 3, 4, 5])
    sl = single_layer_3(sl_input, layers=[3, 4, 3, 2], output_size=3)
    sl.product()
    print('\n')


    """
version 3 test 1
hidden layers structure : [2, 3, 4, 3] 4
Weight Dictionary = OrderedDict([('w1', array([[1, 3, 7],
       [2, 5, 2]])), ('w2', array([[9, 0, 5, 7],
       [5, 4, 5, 4],
       [1, 6, 4, 5]])), ('w3', array([[7, 4, 9],
       [6, 3, 5],
       [5, 1, 6],
       [4, 5, 8]]))])
Bias Dictionary = OrderedDict([('b1', array([[2, 1, 3]])), ('b2', array([[3, 5, 2, 2]])), ('b3', array([[2, 0, 3]]))])
[[0.99908895 0.99999917 0.99999917]]
[[0.99999998 0.99999969 0.99999989 0.99999998]]
[[1.         0.99999774 1.        ]]


version 3 test 2
hidden layers structure : [5, 3, 4, 3, 2, 3] 6
Weight Dictionary = OrderedDict([('w1', array([[8, 1, 5],
       [0, 0, 4],
       [9, 9, 9],
       [0, 3, 6],
       [2, 0, 7]])), ('w2', array([[8, 7, 3, 5],
       [7, 3, 3, 2],
       [0, 8, 6, 5]])), ('w3', array([[8, 1, 9],
       [8, 5, 2],
       [4, 7, 1],
       [4, 3, 5]])), ('w4', array([[5, 6],
       [6, 9],
       [2, 4]])), ('w5', array([[6, 6, 8],
       [3, 4, 7]]))])
Bias Dictionary = OrderedDict([('b1', array([[9, 2, 1]])), ('b2', array([[5, 3, 3, 8]])), ('b3', array([[1, 2, 3]])), ('b4', array([[3, 3]])), ('b5', array([[2, 3, 4]]))])
[[1. 1. 1.]]
[[1.         1.         0.99999969 1.        ]]
[[1.         0.99999998 1.        ]]
[[0.99999989 1.        ]]
[[0.9999833  0.99999774 0.99999999]]
"""

