import inspect
import numerical_differential_1 as nf1
import numpy as np
import matplotlib.pylab as plt

"""
NUMERICAL_DIFFERENTIAL_2.

"""

if __name__ == '__main__':
    for i in inspect.getmembers(nf1, inspect.isfunction):
        print(i)
        """
('numerical_differential', <function numerical_differential at 0x000001F361644558>)
('partial_differential_1', <function partial_differential_1 at 0x000001F36168A9D8>)
('partial_differential_2', <function partial_differential_2 at 0x000001F36168A948>)
('partial_differential_3', <function partial_differential_3 at 0x000001F36FEBF5E8>)
('partial_gradient', <function partial_gradient at 0x000001F36FFAF048>)
('quadratic_1', <function quadratic_1 at 0x000001F36FF508B8>)
-> what are the arguments of partial_gradient?
        """
    print(inspect.getsource(nf1.partial_gradient))
    """
def partial_gradient(function, point, delta=1e-4, verbose=False):
    
parametres:
function : that which we want to get gradients.
point : at which we want to calculate. doesn't need to input as numpy array, float form as default (the function will handle it)
delta: infinitesimal value. 1e-4 for default.
verbose : prints gradient vector if needed. False for default.

return : gradient list, type : array. e.g. [1, 2, 3, 4, 5].

    point = np.array(point).astype(np.float)
    delta_array = np.zeros_like(point)
    gradient_list = []
    for i in range(len(point)):
        delta_array[i] = delta
        gradient = (function(point + delta_array) - function(point - delta_array)) / (2*delta)
        gradient_list.append(gradient[i])
        delta_array[i] = 0
    if verbose:
        print('gradient result =', gradient_list)
    return gradient_list
    """
    # if sourcecode is too long, and we just need to check what are the arguments

    print(inspect.signature(nf1.partial_gradient)) # (function, point, delta=0.0001, verbose=False)

    # We'd like to visualize gradients. the code is referred from https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch04/gradient_2d.py
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X.flatten()
    Y.flatten()
    #print(np.array([X, Y]))

    grads = nf1.partial_gradient(function=nf1.quadratic_1, point=[X, Y])

    plt.figure()
    plt.quiver(X, Y, -grads[0], -grads[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
    

    
        
