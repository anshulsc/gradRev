
import random
from gradrev import engine
from gradrev import nn

draw_dot = nn.draw_dot
Value = engine.Value

# Implementation of backprop on neuron 

# inputs 
x1 = Value(2.0)
x2 = Value(0.0)

# weights 
w1 = Value(-3.0)
w2 = Value(1.0)

# bias of neuron
b = Value(6.8813735870195432)

# x1w1 + x2w2 + b
x1w1 = x1*w1 
x2w2 = x2*w2 
x1w1x2w2 = x1w1 + x2w2 
n = x1w1x2w2 + b 
o = n


o.backward()
dot = draw_dot(o)
dot.render()


