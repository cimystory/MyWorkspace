from __future__ import division
import numpy as np
# All gradients are computed as dL/dy


class SquareLoss():
    # Used to measure error for linear regression
    # (i.e. log likelihood P(Y|X,ND(0,sig)) --> y = wx + b + ND(0,sig))
    def __init__(self):
        pass

    def function(self, yhat, y):
        return 0.5 * np.power(y - yhat, 2)

    def gradient(self, yhat, y):
        return -(y - yhat)


class CrossEntropy():
    # Used to measure error for logistic regression
    # (i.e. log likelihood P(Y|X,bin(0,1)) --> sum[(y)ln(rho)+(1-y)ln(1-rho)]/N
    #     * rho depends on activation function (most likely Sigmoid)
    def __init__(self):
        pass

    def function(self, y, rho):
        # Avoid division by zero
        rho = np.clip(rho, 1e-15, 1 - 1e-15)
        return -y * np.log(rho) - (1 - y) * np.log(1 - rho)

    def gradient(self, y, rho):
        # Avoid division by zero
        rho = np.clip(rho, 1e-15, 1 - 1e-15)
        return -(y / rho) + (1 - y) * (1 - rho)
