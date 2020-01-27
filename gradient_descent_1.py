"""
Gradient Descent 1.

This is an extension of basic_neuron_practice.py. refer pg 132 of 밑바닥부터 시작하는 딥러닝.

"""

import inspect, os

for i in os.listdir(os.curdir):
   print(i)
    
import basic_neuron_practice as bnp

#print('\n Source code of Basic neuron \n',inspect.getsource(bnp.Basic_neuron))
# See that this function has scalar for rescale. Our extension is to modify it into vectors.
import numerical_differential_1 as nf1

#print('list of functions in numerical_differential_1\n')
for i in inspect.getmembers(nf1, inspect.isfunction):
    print(i)
    """
('numerical_differential', <function numerical_differential at 0x000001DBF8C4FEE8>)
('partial_differential_1', <function partial_differential_1 at 0x000001DBF8C5F558>)
('partial_differential_2', <function partial_differential_2 at 0x000001DBF8C5F5E8>)
('partial_differential_3', <function partial_differential_3 at 0x000001DBF8C5F678>)
('partial_gradient', <function partial_gradient at 0x000001DBF8C5F708>)
('quadratic_1', <function quadratic_1 at 0x000001DBF8C5F798>)
    """
# we are going to use function quadratic_1, partial_gradient.

#print('\n Source code of partial_gradient \n',inspect.getsource(nf1.partial_gradient))
#print('\n Source code of quadratic_1 \n',inspect.getsource(nf1.quadratic_1))

"""
The basic concept of gradient descent is simple : in analytics, for a well-defined smooth function F, its local extreme values are occured at which its derivative becomes 0.
For example, suppose that  F = pow(x, 3) +9 * pow(x, 2) -9 * x + 7, then its local extreme values are occured at x = -3±2*sqrt(3), remind that

Now let us imagine a smooth polynomial graph, and we took a arbitrarily given point P, at which is nearby local extreme point.

we can calculate gradient at P, i.e. dF(P). and set a constant L, which we are going to call 'learning rate'. If P is heading to a local extreme point, then its gradient will be dimished into 0.
thus if L*dF(P) converges to 0, its vicinity will be local extreme point. What we need to iterate is x_n+1 = x_n + L*dF(P(x_n)).
"""
def basic_gradient_descent_1(function, point, delta=1e-4, learning_rate=1e-2, steps=1000):
    for i in range(steps):
        bgd1_grad = nf1.partial_gradient(function, point, delta)
        bgd1_grad = np.array(bgd1_grad)
        point -= learning_rate * bgd1_grad
        print(f'step {i} = {bgd1_grad}')
    return point

#we maybe able to improve it.
def basic_gradient_descent_2(function, point, delta=1e-4, learning_rate=1e-2, steps=1000, check=False):
    for i in range(steps):
        bgd1_grad = np.array(nf1.partial_gradient(function, point, delta, verbose=check))
        point -= learning_rate * bgd1_grad
    return point

import numpy as np
"""
def upgrade_bonus(damage):
    counter = 0
    while True:
        if 10*counter - 10 < damage < 10*counter + 5:
            break
        else:
            counter += 1
    return counter

def dps(initial_damage, initial_period, upgrade_level, delay_decrease=True):
    if delay_decrease:
        x = (initial_damage + upgrade_bonus(damage=initial_damage) * upgrade_level) * pow(1.05, upgrade_level) / initial_period
    else:
        upgrade_bonus_prime = round(((pow(1.05, 10) - 1) / 10) * initial_damage + pow(1.05, 10) * upgrade_bonus(damage=initial_damage), 0)
        #print('rescaled upgrade bonus = ', upgrade_bonus_prime)
        x = (initial_damage + upgrade_bonus_prime * upgrade_level) / initial_period
    return x
test components.
"""

#let us make a simple neural network.

# version 1

import loss_function_1 as lf1
import activation_functions_module as afm

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        x = np.array(x).astype(np.float)
        return np.dot(x, self.W)

    def loss(self, x, t):
        x = np.array(x).astype(np.float)
        t = np.array(t).astype(np.float)
        
        y = afm.softmax(self.predict(x))
        loss = lf1.loss_cross_entropy(y, t)

        return loss
        # version 2.
    def dW(self, x, y, delta=1e-4):
        H = np.zeros_like(self.W)
        #print('delta sparse matrix check =', H)
        gradient_Weight = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                H[i][j] = delta
                W_forward = self.W + H
                W_backward = self.W - H
                loss_f = lf1.loss_cross_entropy(afm.softmax(np.dot(x, W_forward)),y)
                loss_b = lf1.loss_cross_entropy(afm.softmax(np.dot(x, W_backward)),y)
                gradient = (loss_f - loss_b) / (2 * delta)
                #print('current gradient = ', gradient)
                H[i][j] = 0
                gradient_Weight[i][j] = gradient
        #print('gradient_Weight matrix check =', gradient_Weight)
        return gradient_Weight

# this is kind of hardcoding but I don't understand at all about using self.predict and self.loss to figure out gradient matrix for weight.
                
                
        
        

def polynomial_1(coefficients, point):
    """
Basic polynomial function on R2 real space.
parametres:
coefficients : array. You need to input as python built-in array like [3, 67, 1, 3, 2]. the array will be input as ascending order 3 + 67x + x^2 + 3x^3 + 2x^4.
point : 
    """
    if coefficients[len(coefficients)-1] == 0:
        raise ValueError('the highest degree shouldn\'t have zero as coefficient')
    polysum = 0
    for i in enumerate(coefficients):  # if coefficients = [10, 20, 30, 40, 50], then it will return (0, 10), (1, 20), (2, 30), (3, 40), (4, 50). where left side is index, right side is item.
        polysum += i[1] * pow(point, i[0])
    return polysum

import matplotlib.pyplot as plt

# it seems that we can improve this polynomial. we want to print like 3 + 67x + x^2 + 3x^3 + 2x^4.

def polynomial_2(coefficients, point, polyform=False):
    """
Basic polynomial function on R2 real space.
parametres:
coefficients : array. You need to input as python built-in array like [3, 67, 1, 3, 2]. the array will be input as ascending order 3 + 67x + x^2 + 3x^3 + 2x^4.
point : 
    """

    string_expression = ''
    if coefficients[len(coefficients)-1] == 0:
        raise ValueError('the highest degree shouldn\'t have zero as coefficient')
    polysum = 0
    for i in enumerate(coefficients):  # if coefficients = [10, 20, 30, 40, 50], then it will return (0, 10), (1, 20), (2, 30), (3, 40), (4, 50). where left side is index, right side is item.
        polysum += i[1] * pow(point, i[0])
        string_expression += f'{i[1]} * x^{i[0]} + '
    if polyform:
        print('polynomial form =',string_expression[:-3].replace(' * x^0',''))
    else:
        string_expression = None
    return polysum


# this code is referred from deep-learning-from-scratch-master\deep-learning-from-scratch-master\common\gradient.py
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print('what is input f?', f)
    print('what is input x? =', x)
    while not it.finished:
        idx = it.multi_index
        print('what is idx? =', idx)
        tmp_val = x[idx]
        print(f'tmp_val=x[idx] ', tmp_val)
        x[idx] = float(tmp_val) + h
        print(f'x[idx] = float(tmp_val)+h =', x[idx])
        print(f'x[idx]-tmp_val =', x[idx] - tmp_val)
        fxh1 = f(x) # f(x+h)
        print('fxh1=f(x) check=', fxh1)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

if __name__ == '__main__':

    
    np.random.seed(121)
    initial_point_1 = [3, 4]
    #function_1 = basic_gradient_descent_1(function=nf1.quadratic_1, point=initial_point_1, steps=1000)
    #function_2 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1)
    #print(function_2) # [5.04890207e-09 6.73186943e-09]

    # what if we differ learning rate?
    #function_3 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1, learning_rate=0.1)
    #print('in case of learning rate=0.1', function_3) # [5.3799827e-21 5.1880768e-21] see that function_3 is better result than function_2.

    # and we can even deliberately diverge it.

    #function_4 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1, learning_rate=10)
    #print('in case of learning_rate=10', function_4) # [ 1.91613251e+13 -1.26893162e+12]

    # applying this gradient descent function, we can modify weights and biases as well.
    """
    print(upgrade_bonus(damage=40))
    print(upgrade_bonus(damage=35))
    print(upgrade_bonus(damage=16))
    print(upgrade_bonus(damage=15))
    print(upgrade_bonus(damage=25))
    print(upgrade_bonus(damage=401))
    print(round(9.0312,0))
    print(round(4.5555,0))
    print(round(4.4444444,0))

    #print('ultralisk dps ram attack =', dps(initial_damage=95, initial_period = 5/3, upgrade_level=5, delay_decrease=False))

    for i in range(11):
        x_1 = dps(initial_damage=95, initial_period = 5/3, upgrade_level=i, delay_decrease=False)
        x_2 = dps(initial_damage=40, initial_period = 0.8608, upgrade_level=i)
        print(f'ultralisk ram attack dps at upgrade level {i} =', x_1)
        print(f'ultralisk kaiser blades attack dps at upgrade level {i} =', x_2)
        print(f'ultralisk ram/kaiser blades dps rate at upgrade level {i} =', x_1/x_2)
    """        
    net = SimpleNet()
    print('initialised Weights =',net.W)# [[-0.21203317 -0.28492917 -0.57389821] [-0.44031017 -0.33011056  1.18369457]]
    print('the size of net.W is =', net.W.shape)
    #for i in range(net.W.shape[0]):
    #   for j in range(net.W.shape[1]):
    #       print(net.W[i][j])

    ipt = [6, 9]
    #prd = net.predict(ipt)
    #print(f'preidcted weights with {ipt} =', prd) # [6, 9] = [-5.23499049 -4.68057002  7.20986188]
    #print('which index is biggest?', np.argmax(prd), prd[np.argmax(prd)]) # 2 7.20986187635347
    
    opt = [0, 0, 1]
    #ans = net.loss(x=ipt, t=opt)
    #print('ans = ', ans)

    # let us test nf1.partial_gradient
    #print(polynomial_1(coefficients=[1, 2, 3, 4, 5], point=5))  #3711
    #polynomial_2(coefficients=[1, 2, 3, 4, 5], point=5, polyform=True) # polynomial form = 1 + 2 * x^1 + 3 * x^2 + 4 * x^3 + 5 * x^4

    """
    xs = [i/100 for i in range(-1000, 1000)]
    polynomial_tracker = []
    for x in xs:
        polynomial_tracker.append(polynomial_1(coefficients=[0, 0, 1], point=x))
    plt.plot(xs, polynomial_tracker)
    plt.show()
    """

    #dW = net.dW(x=ipt, y=opt)
    #print(dW)
    
    """
[[ 2.36273946e-05  4.11336811e-05 -6.47610774e-05]
 [ 3.54410949e-05  6.17005278e-05 -9.71416210e-05]]
    """
    f_2 = lambda b : net.loss(ipt, opt)
    dW_2 = numerical_gradient(f=f_2, x=net.W)
    print(dW_2)  # same result.

