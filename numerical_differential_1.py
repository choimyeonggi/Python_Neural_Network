"""
Numerical differential.

Since it is impossible to obtain exact analytic form of numerical functions in computer, using numerically approximated form is natural.

In order to differentiate a well defined function fn(x), then for given h=1e-4,

fn(x)/dx = (fn(x+h) - fn(x))/h.

in other words,

fn(x)/dx = (fn(x-h) - fn(x))/h.

the former, which we know very well, is called front differentiation, the latter is called backward differentiation.

in order to improve accuracy, use two points : x+h, x-h. then the differential will be

fn(x)/dx = (fn(x+h)-fn(x-h))/(2*h).
"""

import numpy as np

def numerical_differential(function, point):
    """Suppose that 'function' variable is based on R2 space, and well defined smooth.
refer from DeepLearning/ch04/azerates20191227d.py
    """
    h = 1e-4
    return (function(point+h) - function(point-h))/(2*h)

"""
Imagine that we are going to handle multi-variable function, and we want to get partial differential.

Set
def function_1(x, y):
    return pow(x, 2) + pow(y, 2), which means function(x, y) = x^2 + y^2.

then partial differential is like

function_1/dx = 2x
function_1/dy = 2y

in other word we can express function_1 like this

def function_1(x:array):
    return np.sum(pow(x,2)) -> suppose that x is a numpy array, i.e. x=np.array([1,2,3,4, ... ]).
    then pow(x,2) means an array that has powered element w.r.t x. i.e. pow(x,2)=np.array([1, 4, 9, 16, ... ])

in order to calculate partial differentail w.r.t x[0], suppose that x=[x_1, x_2] then the formula shall be

(function([x_1, x_2] + [h, 0]) - function([x_1, x_2] - [h, 0])) / (2 * [h, 0])

then partial differential for x[1] is clear :

(function([x_1, x_2] + [0, h]) - function([x_1, x_2] - [0, h])) / (2 * [0, h])

now we got a clue for coding interpretation : h=1e-4 -> [ ... h ... ]

let us make prototype for partial differential.
"""
# version 1.
def partial_differential_1(function, point, index, delta=1e-4):
    """
suppose that point=[x1, x2], and index=1 (natural number). We'd like to return (function([x1, x2]+[0, delta]) - function([x1, x2] - [0, delta]))/(2*[0, delta]).
then create an empty array that has same length w.r.t point

    """
    point = np.array(point).astype(np.float)  # for some unknown reason, we must transform array into float types.
    delta_array = np.zeros_like(point)
    delta_array[index] = delta
    print('initialised delta_array =', delta_array)
    print('point to be differentiated =', point)
    print(point+delta_array, point-delta_array)
    #partial differential at index.
    partial_gradient = (function(point + delta_array) - function(point - delta_array)) / (2*delta)
    return partial_gradient

# Can we script more generalised version of partial_differential_1?

#version 2.
def partial_differential_2(function, point, delta=1e-4):
    """
Suppose that point=[x1, x2], then we'd like to return each partial differential set, i.e. tuple. How can we iterate it?

It seems like delta_array[index], and partial gradient must be modified.
    """
    point = np.array(point).astype(np.float)
    delta_array = np.zeros_like(point)
    """
Iteration 1: delta_array[0] = delta -> partial_gradient[0] = (function(point + delta_array) - function(point - delta_array)) / (2*delta) -> delta_array[0] = 0
Iteration 2: delta_array[1] = delta -> partial_gradient[1] = (function(point + delta_array) - function(point - delta_array)) / (2*delta) -> delta_array[1] = 0
Iteration 3: delta_array[0] = delta -> partial_gradient[0] = (function(point + delta_array) - function(point - delta_array)) / (2*delta) -> delta_array[0] = 0
...

-> create a empty list 'partial gradient'
    """
    partial_gradient = []
    # iteration length : length of point.
    for i in range(len(point)):
        delta_array[i] = delta
        gradient = (function(point + delta_array) - function(point - delta_array)) / (2*delta)
        partial_gradient.append(gradient)
        delta_array[i] = 0
    return partial_gradient

# what if we want to return gradient tuple?

def partial_differential_3(function, point, delta=1e-4, gradient_set=False):
    point = np.array(point).astype(np.float)
    delta_array = np.zeros_like(point)
    partial_gradient = []
    for i in range(len(point)):
        delta_array[i] = delta
        gradient = (function(point + delta_array) - function(point - delta_array)) / (2*delta)  # the result is array.
        if gradient_set:
            partial_gradient.append(gradient[i]) # -> we can make this as option parameter.
        else:
            partial_gradient.append(gradient)
        delta_array[i] = 0
    partial_gradient = tuple(partial_gradient)
    return partial_gradient

def partial_gradient(function, point, delta=1e-4, verbose=False):
    """
parametres:
function : that which we want to get gradients.
point : at which we want to calculate. doesn't need to input as numpy array, float form as default (the function will handle it)
delta: infinitesimal value. 1e-4 for default.
verbose : prints gradient vector if needed. False for default.

return : gradient list, type : array. e.g. [1, 2, 3, 4, 5].
    """
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
    return np.array(gradient_list)


def quadratic_1(points):
    """
    points : array.
simple quadratic. no specific coefficients. np.sum(points) = x1^2 + x2^2 + x3^2 + ...
->
quadratic =[]
for i in range(len(points)):
    point_powered = power(points[i], 2)
    quadratic.append(point_powered)

    """
    quadratic =[]
    for i in range(len(points)):
        point_powered = pow(points[i], 2)
        quadratic.append(point_powered)

    return np.array(quadratic)


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7]
    x_2 = quadratic_1(points=x)
    print(x_2) # [1, 4, 9, 16, 25, 36, 49]
    y = np.array(x)
    y_2 = quadratic_1(points=y)
    print(y_2) # [1, 4, 9, 16, 25, 36, 49]

    diff_1 = partial_differential_1(function=quadratic_1, point=[3, 4, 5], index=1)
    print('diff_1 =', diff_1) # [0. 8. 0.]

    z = np.zeros_like(x)
    print(z) # [0 0 0 0 0 0 0]
    z[1] = 1
    print(z) # [0 1 0 0 0 0 0]

    # let us test partial_differential_2.
    
    diff_2 = partial_differential_2(function=quadratic_1, point=[3, 4, 5])
    print('diff_2 =', diff_2) # [array([6., 0., 0.]), array([0., 8., 0.]), array([ 0.,  0., 10.])]
    print('diff_2[1] =', diff_2[1]) # [0. 8. 0.] == diff_1

    # what if we differ delta by 1e-6? hope that the result won't be changed.

    print(partial_differential_2(function=quadratic_1, point=[3, 4, 5], delta=1e-6))  # [array([6., 0., 0.]), array([0., 8., 0.]), array([ 0.,  0., 10.])]

    # Can we occur underflow?

    print(partial_differential_2(function=quadratic_1, point=[3, 4, 5], delta=1e-8))   # maybe underflow? [array([5.99999996, 0.        , 0.        ]), array([0.        , 7.99999995, 0.        ]), array([0.        , 0.        , 9.99999994])]

    print(partial_differential_2(function=quadratic_1, point=[3, 4, 5], delta=1e-10))  # [array([6.0000005, 0.       , 0.       ]), array([0.        , 8.00000066, 0.        ]), array([ 0.        ,  0.        , 10.00000083])]

    # delta=1e-4 is enough. never the more, the less.

    # partial_differential_3 test.

    print('partial_differential_3 test, gradient_set=False =',partial_differential_3(function=quadratic_1, point=[1, 2, 3, 4, 5], delta=1e-2, gradient_set=False)) # (array([2., 0., 0., 0., 0.]), array([0., 4., 0., 0., 0.]), array([0., 0., 6., 0., 0.]), array([0., 0., 0., 8., 0.]), array([ 0.,  0.,  0.,  0., 10.]))
    print('partial_differential_3 test, gradient_set=True =',partial_differential_3(function=quadratic_1, point=[1, 2, 3, 4, 5], delta=1e-2,  gradient_set=True)) # (2.0000000000000018, 3.999999999999937, 5.999999999999872, 7.9999999999998295, 9.999999999999787)

    # partial gradient test

    partial_gradient(function=quadratic_1, point=[1, 2, 3, 4, 5, 6, 7], delta=1e-2) # [2.0000000000000018, 3.999999999999937, 5.999999999999872, 7.9999999999998295, 9.999999999999787, 11.999999999999744, 13.999999999999702]

    matrix_1 = partial_gradient(function=quadratic_1, point=np.array([[1, 2, 3], [4, 5, 6]]))
    print(matrix_1)
    matrix_2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(matrix_2)
    
