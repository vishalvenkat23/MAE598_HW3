# A simple example of using PyTorch for gradient descent
import numpy
import torch
# from scipy import exp as exp
# from scipy import gradient
from torch.autograd import Variable

psatw = 10 ** (8.07131 - (1730.60 / (20 + 233.426)))
psatd = 10 ** (7.43155 - (1554.679 / (20 + 240.337)))
x1knob = numpy.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
x2knob = 1 - x1knob
p = numpy.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

x = Variable(torch.tensor([1.0, 0.0]), requires_grad=True)
x = x.detach().numpy()

# Fix the step size
a = 0.001

# Start gradient descent
for i in range(1000):  # change the termination criterion
    for i in range(0, len(x1knob)):
        loss = (x1knob[i] * numpy.exp(
            x[0] * ((x[0] * x2knob[i]) / (x[0] * x1knob[i] + x[1] * x2knob[i])) ** 2) * psatw +
                (x2knob[i] * numpy.exp(
                    x[1] * ((x[0] * x1knob[i]) / (x[0] * x1knob[i] + x[1] * x2knob[i])) ** 2) * psatd)
                - p[i]) ** 2
        loss.backward()
    x.grad.scipy()
    # no_grad() specifies that the operations within this context are not part of the computational graph, i.e., we don't need the gradient descent algorithm itself to be differentiable with respect to x
    with torch.no_grad():
        x -= a * x.grad

        # need to clear the gradient at every step, or otherwise it will accumulate...
        x.grad.zero_()

print(x.data.numpy())
# print(loss.data.numpy())
# Define a loss
for i in range(len(x1knob)):
    p_error = (x1knob[i] * numpy.exp(
        x[0] * ((x[0] * x2knob[i]) / (x[0] * x1knob[i] + x[1] * x2knob[i])) ** 2) * psatw + (x2knob[i] * numpy.exp(
        x[1] * ((x[0] * x1knob[i]) / (x[0] * x1knob[i] + x[1] * x2knob[i])) ** 2) * psatd))
    error = p_error - p[i]
    print(error)
