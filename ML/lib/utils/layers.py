import math
import copy
import numpy as np
from activation_funcs import Sigmoid, ReLU, LeakyReLU, TanH, SoftMax


# Layer subclass
class Layer(object):
    # Layer SuperClass
    def set_input_shape(self, shape):
        self.input_shape = shape

    def get_layer_name(self):
        return self.__class__.__name__

# Types of layers sub-sub-class


class Dense(Layer):
    """
    fully-connected NN layer --> subclass of Layer SuperClass.

    Inputs:
    -----------
    n_units:     [int] The number of nodes in the layer.
    input_shape: [tuple] The expected input shape of the layer. single digit for Dense.
                 -  Must be specified if it is the first layer in the network.
    """

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.initialized = False
        self.trainingState = True
        self.W = None
        self.wb = None  # bias term

        self.input_shape = input_shape
        self.n_units = n_units

    def initialize(self, optimizer):
        nodes_in = self.input_shape[0]
        nodes_out = self.n_units

        bound = 1 / math.sqrt(nodes_in)
        # initialize weights to be Normal distribution ND(1/sqrt(n_in))
        self.W = np.random.normal(-bound, bound, (nodes_in, nodes_out))
        self.wb = np.zeros((1, nodes_out))  # bias term
        # return a copy of user specified optimizer
        self.W_opt = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def forward_pass(self, X, trainingState=True):
        # linear regression
        self.layer_input = X
        return X.dot(self.W) + self.wb

    def backward_pass(self, prop_grad):
        # Propagate gradients backwards through NN to update layer weights

        W = self.W  # store weights in memory

        if self.trainingState:
            # Calculate gradient w.r.t layer weights
            grad_w = np.dot(self.layer_input.T, prop_grad)
            grad_wb = np.sum(prop_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.wb = self.wb_opt.update(self.wb, grad_wb)

        # Return propagated gradient for next layer
        # Calculated based on the weights used during the forward pass
        prop_grad = np.dot(prop_grad, W.T)

        return prop_grad

    def backwardQuery(self, layer_output):
        layer_output = np.dot(layer_output, self.W.T)

        return layer_output

    def output_shape(self):
        return (self.n_units,)


class Dropout(Layer):
    """A layer that randomly sets a fraction of previous layer output units to zero.

    Inputs:
    -----------
    p: [float]  The probability that unit x is set to zero.
    """

    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainingState = True

    def forward_pass(self, X, trainingState=True):
        c = (1 - self.p)
        if trainingState:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, prop_grad):
        return prop_grad * self._mask

    def backwardQuery(self, layer_output):
        return layer_output

    def output_shape(self):
        return self.input_shape


class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Inputs:
    -----------
    name: string
        The name of the activation function that will be used.
    """
    activation_functions = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'softmax': SoftMax,
        'leaky_relu': LeakyReLU,
        'tanh': TanH,
    }

    def __init__(self, name):
        self.activation_name = name
        self.activation = self.activation_functions[name]()
        self.trainingState = True

    def forward_pass(self, X, trainingState=True):
        self.layer_input = X
        return self.activation.function(X)

    def backward_pass(self, prop_grad):
        return prop_grad * self.activation.gradient(self.layer_input)

    def backwardQuery(self, layer_output):
        return self.activation.ifunction(layer_output)

    def output_shape(self):
        return self.input_shape
