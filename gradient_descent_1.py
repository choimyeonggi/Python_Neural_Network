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
# we are going to use function quadratic_1 
