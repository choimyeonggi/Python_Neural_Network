"""
LOSS FUNCTION

2020 01 16 JJH
"""

# 1. Sum of suqares for error.

"""
Let us suppose that the data which we are going to handle have answer labels(like MNIST), and we have a neural network, a black box which we want to predict number graphics as precise as possible.
When we are to estimate the performance of the black box(From now on, we are going to call neural network as black box). We may be able to measure 'performance' with answers and predicted values.

Let y_i be predicted values by black box, a_i be answers. Then E, the total sum of error is like below

E = SUM(pow((y_i - a_i), 2)).

Note that there is no unit for errors. What the important is reducing these 'errors' as much as possible. I.e. they are RELATIVE. And since y_i are dependent to WEIGHTS AND BIASES, the problem is clear:
Can wer optimize them where the errors become minimal?

"""
import os, sys
import numpy as np
import pickle

dataset_path = os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'dataset').replace('\\', '/')
sys.path.append(dataset_path)
sample_weight_path = os.path.join('..', '..', '..', 'PycharmProjects', 'DeepLearning', 'ch03', 'sample_weight.pkl').replace('\\', '/')

from mnist import load_mnist
from neural_network_2_mnist import mini_batch, init_network

def loss_sum_of_squares_for_error(x, y):
    """ Note that parametres, x and y, are must be numpy array. """
    return np.sum((x-y)**2)

# 2. cross entropy error.

"""
The cross entropy between two probability distributions p and q over the same underlying set of event measures the average of bits needed to identify an event
drawn from the set if a coding scheme used for the set is optimized for an estimated probability distribution q, rather than the true distribution p.

The cross entropy of the distribution q relative to a distribution p over given set is defined as follows : H(p, q) = -E_p [log q].

In information theory, the Kraft-McMilan theorem establishes that any directly decodable coding scheme for coding message to identify one value x_i, out of set of possibilities {x_1 ~ x_n}
CAN BE SEEN AS REPRESENTING an IMPLICIT PROBABILTY DISTRIBUTION, q(x_i) = pow(0.5, l_i) over {x_1 ~ x_n}, where l_i is the length of the code for x_i in bits.
Therefore, CROSS ENTROPY can be interpreted as the expected message length  per datum, when a wrong distribution q is assumed while the data actually follows a distribution p.
That is, why the expectation is taken over the true probability distribution p and not q. Indeed, the expected message length under the true distribution p is :

E_p [l] = -E_p [log_2 q(x)] = -SUM(p(x_i) * log_2 q(x_i))

the function is like below : (referred from DeepLearning/ch04/azerates20191227b_cross_entropy.py
def _cross_entropy(y_pred, y_true, delta=1e-7):
    # Since log function is not defined at 0, add a very small value to avoid error.
    # np.log(0) returns RuntimeWarning: divide by zero encountered in log error, prints -inf.

    return -np.sum(y_true * np.log(y_pred + delta))

def cross_entropy(y_pred, y_true, delta=1e-7):
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true, delta)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true, delta) / len(y_pred)
    return ce

can we improve this?

def cross_entropy(predicted, trues, delta=1e-7):
    if predicted.ndim == 1:
        y = -np.sum(trues * np.log(predicted + delta))
    elif predicted.ndim == 2:
        y = -np.sum(trues * np.log(predicted + delta)) / len(predicted)
    return y

    now we don't need 2 functions to run cross entropy.

"""
def loss_cross_entropy(predicted, trues, delta=1e-7, check=False):
    if check:
        print('loss_cross_entropy, input length is =', len(predicted))
        print('the dimension of input data is =', predicted.ndim)
    if predicted.ndim == 1:
        y = -np.sum(trues * np.log(predicted + delta))
    elif predicted.ndim == 2:
        y = -np.sum(trues * np.log(predicted + delta)) / len(predicted)
    return y

#let us check cross entropy.

def batch_index(train_size, batch_size=10):
    """
Suppose that the target data is big, million, just calculating loss function would take pretty long time, which is not efficient.
Hence we need to sampling data instead of calculate all. (Think that, it is usually difficult to obtain accurate stochastic values by complete enumeration.

parametres:
train size : natural number. with respect to MNIST, x_train.shape[0]=60000.
batch_size : sampling number. if batch_size=10, take 10 random indices.

thus, if batch_index(60000, 10) is equivalent to np.random.choice(60000, 10). 

Return : batch_indices.
    """
    batch_indices = np.random.choice(train_size, batch_size)
    return batch_indices

    

if __name__ == '__main__':
    network_sample = init_network()
    #print(network_sample)
    
    (x_train , y_train), (x_test, y_test) = load_mnist()
    print(y_train[:20]) # [5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7 2 8 6 9]
    y_predicted = mini_batch(network=network_sample, x=x_train, batch_size=10)
    print(y_predicted[:20]) # [5. 0. 4. 1. 9. 2. 1. 3. 1. 4. 3. 5. 3. 6. 1. 7. 2. 8. 6. 9.]

    #now let us measure loss.

    loss_1 = loss_sum_of_squares_for_error(y_train[:20], y_predicted[:20])
    print('loss_1 =',loss_1)  # '0'


    # what if we use another data for it?

    loss_2 = loss_sum_of_squares_for_error(y_train, y_predicted)
    print('loss_2 =',loss_2)  # 68994. what if we divide this by length?
    print('loss_2 mean =',loss_2/len(y_train))  # 1.1499


    # let us measure cross entropy.

    cross_entropy_1 = loss_cross_entropy(predicted=y_predicted, trues=y_train)
    print(f'cross_entropy_1 = {cross_entropy_1}') # -436354.43926602375
    print(f'cross_entropy_1 mean = {cross_entropy_1 / len(y_train)}') # -7.272573987767062

    np.random.seed(1227)
    y_true = np.random.randint(10, size=10) # [4 3 9 7 3 1 6 6 8 8]
    print('RANDOM ARRAY_1 = \n',y_true)

    y_true2 = np.zeros((y_true.size, 10))  # 10 * 10 matrix filled with 0s.
    print('ZERO MATRIX = \n',y_true2)

    for i in range(y_true.size): # y_true.size = 10
        y_true2[i][y_true[i]] = 1  # if i=0, then y_true2[0][y_true[0]] = y_true2[0][4] =1.
    y_true_ohe = y_true2.astype('int')

    print('ZERO MATRIX FILLED 1 BY INDEX OF RANDOM ARRAY_1 = \n',y_true_ohe)  # in order to mark one-hot-encoding, above code will help

    # Use mini_batch for random sampling.

    indices = batch_index(train_size=x_train.shape[0], batch_size=100)
    x_train_batch = x_train[indices]
    y_train_batch = y_train[indices]

    # let us check them.

    #print(f' x and y train batches ={x_train_batch, y_train_batch}')


    # and another cross entropy would be
    cross_entropy_2 = loss_cross_entropy(predicted=mini_batch(network=network_sample, x=x_train_batch, batch_size=10), trues=y_train_batch)
    print('cross_entropy_2 = ', cross_entropy_2) # -832.4572481130118
    print('cross_entropy_2 mean =', cross_entropy_2 / len(y_train_batch)) # -8.324572481130119

    

    
    
    
    
    
