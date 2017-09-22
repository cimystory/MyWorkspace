import numpy as np
import sys
#import scipy.special
#scipy.special.logit(y) 
# reference: https://en.wikipedia.org/wiki/Activation_function
class Sigmoid():
    def __init__(self):
        pass
    def function(self,z):
        return 1 / (1 + np.exp(-z))
    def gradient(self,z):
        return self.function(z) * (1-self.function(z))
    def ifunction(self,y):
        # logit
        y = np.clip(y, 1e-15, 1-1e-15)
        return np.log(y) -  np.log(1 - y)

class SoftMax():
    # i.e. Weighted sigmoid
    def __init__(self):
        pass
    def function(self,z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    def gradient(self,z):
        return self.function(z) * (1-self.function(z))
    def ifunction(self,y):
        # multi-nomial logit
        y = np.clip(y, 1e-15, 1-1e-15)
        return np.log(y) - np.log(1 - y)

class TanH():
    def __init__(self):
        pass
    def function(self,z):
        return 2 / (1 + np.exp(-2 * z)) - 1
    def gradient(self,z):
        return 1 - np.power(self.function(z), 2)
    def ifunction(self,y):
        # arctan
        y = np.clip(y, -1 + 1e-15, 1 - 1e-15)
        return np.arctanh(y) 

class ReLU():
    # z  |  z > 0
    # 0  |  z < 0
    def __init__(self):
        pass
    def function(self, z):
        return np.where(z >= 0, z, 0)

    def gradient(self, z):
        return np.where(z >= 0, z, 0)

class LeakyReLU():
    # z        |  z > 0
    # z*alpha  |  z < 0
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def function(self, z):
        return np.where(z >= 0, z, self.alpha * z)

    def gradient(self, z):
        return np.where(z >= 0, 1, self.alpha)
