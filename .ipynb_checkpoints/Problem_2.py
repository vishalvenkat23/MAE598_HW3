# A simple example of using PyTorch for gradient descent

import torch as t
from torch.autograd import Variable
from numpy import linspace, exp
psatw = 10 ** (8.07131 - (1730.60/(20 + 233.426)))
psatd = 10 ** (7.43155 - (1554.679/(20 + 240.337)))
x1knob = linspace(0, 0.7, 8)
x2knob = 1 - x1knob
# Define a variable, make sure requires_grad=True so that PyTorch can take gradient with respect to this variable
x = Variable(t.tensor([1.0, 0.0]), requires_grad=True)

# Define a loss
loss = (x1knob*exp(x[0]*((x[0]*x2knob)/(x[0]*x1knob + x[1]*x2knob)) ** 2) * psatw + (x2knob*exp(x[1]*((x[0]*x1knob)/(x[0]*x1knob + x[1]*x2knob)) ** 2)* psatd))


# Take gradient
loss.backward()

# Check the gradient. numpy() turns the variable from a PyTorch tensor to a numpy array.
x.grad.numpy()
