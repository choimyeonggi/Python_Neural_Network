"""
WARNING : THESE CODES WERE WRITTEN ON NOTEPAD++ TO EDIT&INTEGRATE THEM RELATIVELY LIGHTER. HENCE PEP MIGHT HAVE NOT BEEN KEPT WELL.
(I'M GOING TO WRITE CODES ON PYCHARM ENVIROMENT FROM NOW ON)


forward propagation 002

Let us assume that the former file 'forward_propagation_001' is stored at same directory, and we have a file 'sample_weight.pkl' which has an information of weights and biases for MNIST.

1. place 'forward_propagation_001.py' where you can import it well
2. place 'sample_weight.pkl' as well. 

We are going to calculate accuracy with neural network we defined.


"""

import pickle
import numpy as np
from dataset.mnist import load_mnist
from ch03.forward_propagation_001 import forward_propagation

# First step. We need to create a neural network that has weights and biases.

def init_network():
	with open('sample_weight.pkl', mode='rb') as fi:
		network = pickle.load(fi)
	return network
"""
pickle package

datum = {
    'name': 'jjh',
    'age': 27,
    'k1': [1, 2.0, 'A0'],
    'k2': {'tel': '010-1234-5678', 'e-mail': 'wilsoninkr@gmail.com'}
}

with open('datum_dict.pkl', mode='wb') as fi:
    pickle.dump(datum, fi)

with open('datum_dict.pkl', mode='rb') as fi:
    reload = pickle.load(fi)

print(reload)

you can dump / load a file with pickle package. it allows us to read and write files as binary. (wb means write binary, rb read binary)

{with open('sample_weight.pkl', mode='rb') as fi} means read 'sample_weight.pkl' as binary.

so {return network} means a neural network itself, consisted of weights and biases matrices.

"""

def accuracy(predicted, answers, ntw, f_prop):
    """
    predicted = data we want to predict. Passing neural network one by one.
    answers = which we want to compare to predicted.
    """
    if predicted.shape[0] == len(answers):
        cnt = 0
        for i in range(len(predicted)):
            comparison = np.argmax(f_prop(ntw, predicted[i]))
            if comparison == answers[i]:
                cnt += 1
    else:
        raise ValueError('Two parameters must have same length.')
    return cnt / len(predicted)

if __name__ == '__main__':
    ntw = init_network()
    print(ntw.keys())  # dict_keys(['b2', 'W1', 'b1', 'W2', 'W3', 'b3'])
    print(ntw['W1'].shape)  # 784 * 50
    print(ntw['W2'].shape)  # 50 * 100
    print(ntw['W3'].shape)  # 100 * 10
    print(ntw['b1'].shape)  # 50 * 1
    print(ntw['b2'].shape)  # 100 * 1
    print(ntw['b3'].shape)  # 10 * 1
	
	"""before applying forward propagation, each size of weighted matrices is very important information, as it determines product possibility.
	
	size(W1) = 784*50
	size(W2) = 50*100
	size(W3) = 100*10. Well defined. row number means the number of variables, while col number is the number of hidden layers.
	
	Since the weighted matrix ends at 3, We can guess that the output will have 10 columns. In classification, it means it will be one of 10 characteristics.
	"""
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print(X_train.shape)  # 60000, 784.
    print(y_train.shape)  # 60000, 1
    print(X_test.shape)  # 10000, 784
    print(y_test.shape)  # 10000, 1.
	
	"""
	
	1. normalize=True : the data are basically in range of 0~255(rgb). When we need to apply functions such as exponential, or something else, normalization is fittest one. (Especially, for exponential functions, it prevents us from overflow) 
	
	2. flatten=True. default. Basically each datum(X_test[i]) is image file, 28*28pixel.
	
	3. one_hot_label=False. default. if True, it means one of them has highest likelihood, array like
	[0, 0, ... , 1 , 0, 0, ... , 0]
	(Since we are going to decide it at last moment, we shouldn't set it True)
	
	Our goal is to predict Test data as accurate as possible. To do that, we need to initialize neural network, which we already have done, input training and test data for it.
	
	"""
    y_pred = forward_propagation(ntw, X_test)
	
	# Now we can calculate accuracy.
	
	pt_count = 0
    for s in range(len(y_pred)):
        if np.argmax(y_pred[s]) == y_test[s]:
            pt_count += 1
    print(pt_count / len(y_pred))
	
	# by this manner, the accuracy is 0.9352, 93.52%. Can we define function for generality?
	"""
	y_test is answer(unsigned 8bit int), and each element of y_pred is np.ndarray.
	So we needed np.argmax() for comparison. See that we used X_test to create predicted label.
	and its size is 10000*784, while y_test is 10000*1. Hence, if X_test.shape[0] == y_test, comparison holds.
	
	def accuracy(X_test, y_test):
        ntw = init_network()
        if X_test.shape[0] == len(y_test):
            cnt = 0
            for i in range(len(X_test)):
                comparis = np.argmax(forward_propagation(ntw, X_test[i]))
                if comparis == y_test[i]
                    cnt += 1
        return cnt / len(X_test)
	
	we may be able to set neural network body(at this point, init_network) be parameter, so is forward propagation.

	def accuracy(X_test, y_test, ntw, f_prop):
        if X_test.shape[0] == len(y_test):
            cnt = 0
            for i in range(len(X_test)):
                comparison = np.argmax(f_prop(ntw, X_test[i]))
                if comparison == y_test[i]
                    cnt += 1
        return cnt / len(X_test)
	
	replace X_test = predicted, y_test = answers.
	"""
        
	# example.
	acr2 = accuracy(X_test, y_test, ntw=init_network(), f_prop=forward_propagation)
	print(acr2)  # 0.9352.
	
