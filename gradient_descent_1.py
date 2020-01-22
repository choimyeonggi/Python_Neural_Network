"""
Gradient Descent 1.

This is an extension of basic_neuron_practice.py. refer pg 132 of 밑바닥부터 시작하는 딥러닝.

"""

import inspect, os

for i in os.listdir(os.curdir):
    print(i)
    
import basic_neuron_practice as bnp

print(inspect.getsource(bnp.Basic_neuron))
# See that this function has scalar for rescale. Our extension is to modify it into vectors.
import numerical_differential_1 as nf1

for i in inspect.getmembers(nf1, inspect.isfunction):
    print(i)
# we are going to use function quadratic_1, partial_gradient.

print(inspect.getsource(nf1.partial_gradient))
print(inspect.getsource(nf1.quadratic_1))

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
        
        z = self.predict(x)
        y = afm.softmax(z)
        loss = lf1.loss_cross_entropy(y, t)

        return loss


if __name__ == '__main__':
    np.random.seed(121)
    initial_point_1 = [3, 4]
    #function_1 = basic_gradient_descent_1(function=nf1.quadratic_1, point=initial_point_1, steps=1000)
    function_2 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1)
    print(function_2) # [5.04890207e-09 6.73186943e-09]

    # what if we differ learning rate?
    function_3 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1, learning_rate=0.1)
    print('in case of learning rate=0.1', function_3) # [5.3799827e-21 5.1880768e-21] see that function_3 is better result than function_2.

    # and we can even deliberately diverge it.

    function_4 = basic_gradient_descent_2(function=nf1.quadratic_1, point=initial_point_1, learning_rate=10)
    print('in case of learning_rate=10', function_4) # [ 1.91613251e+13 -1.26893162e+12]

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
    print('initialised Weights =',net.W)

    ipt = [6, 9]
    prd = net.predict(ipt)
    print(f'preidcted weights with {ipt} =', prd)
    print('which index is biggest?', np.argmax(prd), prd[np.argmax(p)])
    
    opt = [0, 0, 1]
    ans = net.loss(x=ipt, t=opt)
    
    
    
    # now we can calculate gradient. think that Weights are function, and we want to minimize loss.
    def f(W):
        return net.loss(

    
