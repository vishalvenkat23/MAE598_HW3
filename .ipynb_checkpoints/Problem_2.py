import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI
def black_func(x, y):
    return 4 - 2.1 * x ** 2 + ((x ** 4) / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2


