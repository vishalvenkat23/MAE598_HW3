# import sklearn.gaussian_process as gp

# code for trial example
# example of the test problem
from math import sin
from math import pi
from numpy import arange
from numpy import argmax
from numpy.random import normal
from matplotlib import pyplot


# objective function
def objective(x, noise=0.1, X1=arange(-2, 2, 0.1), X0=arange(-3, 3, 0)):
    noise = normal(loc=0, scale=noise)
    # return (x ** 2 * sin(5 * pi * x) ** 6.0) + noise
    return ((4 - 2.1*x[0]**2 + ((x[0]**4)/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2) + noise

# grid-based sample of the domain [0,1]
X_0 = arange(-3, 3, 0.1)
X_1 = arange(-2, 2, 0.1)
# sample the domain without noise
y = [objective(x, 0) for x[0] in X_0 for x[1] in X_1]
# sample the domain with noise
ynoise = [objective(x) for x[0] in X_0 for x[1] in X_1]
# find best result
ix = argmax(y)
print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))
# plot the points with noise
pyplot.scatter(X, ynoise)
# plot the points without noise
pyplot.plot(X, y)
# show the plot
pyplot.show()
