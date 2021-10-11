# A simple example of using PyTorch for gradient descent
import numpy
import torch
import math
# from scipy import exp as exp
# from scipy import gradient
from torch.autograd import Variable

p_satw = float(10 ** (8.07131 - (1730.60 / (20 + 233.426))))
p_satd = float(10 ** (7.43155 - (1554.679 / (20 + 240.337))))
x1_knob = numpy.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
x2_knob = 1 - x1_knob
p = numpy.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

x = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

# Fix the step size
a = 0.001

# Start gradient descent
for i in range(250):  # change the termination criterion
    for i in range(0, len(x1_knob)):
        loss = ((((x1_knob[i] * torch.exp(x[0] * ((x[1] * x2_knob[i]) / (x[0] * x1_knob[i] + x[1] * x2_knob[i])) ** 2)) *
                 p_satw) + (x2_knob[i] * torch.exp(x[1] * ((x[0] * x1_knob[i]) / (x[0] * x1_knob[i] + x[1] * x2_knob[i]))
                                                   ** 2) * p_satd)) - p[i]) ** 2
        loss.backward()
        # print(loss.data.numpy())
    x.grad.numpy()
    # no_grad() specifies that the operations within this context are not part of the computational graph,
    #     i.e., we don't need the gradient descent algorithm itself to be differentiable with respect to x
    with torch.no_grad():
        x -= a * x.grad
        # need to clear the gradient at every step, or otherwise it will accumulate...
        x.grad.zero_()

print(x.data.numpy())
print(loss.data.numpy())

for i in range(0, len(x1_knob)):
    p_error = ((x1_knob[i] * math.exp(x[0] * ((x[1] * x2_knob[i]) / (x[0] * x1_knob[i] + x[1] * x2_knob[i])) ** 2)) * p_satw
               + (x2_knob[i] * math.exp(x[1] * ((x[0] * x1_knob[i]) / (x[0] * x1_knob[i] + x[1] * x2_knob[i])) ** 2) * p_satd))
    error = float(p[i] - p_error)
    print("Data of P from table ", p[i], "P error = ", p_error, "Error is ", error)
