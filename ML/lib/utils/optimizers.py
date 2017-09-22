import numpy as np
# Optimizers for models that use gradient methods for finding the
# weights that minimizes the loss.
#resource: http://sebastianruder.com/optimizing-gradient-descent/index.html

class GradientDescent():
    def __init__(self,learningRate=0.01,momentum=0):
        self.learningRate = learningRate
        self.momentum     = momentum
        self.w_update     = np.array([])

    def update(self, w, grad_w):
        # For first update
        if self.w_update.any():
            self.w_update = np.zeros(w.shape)
        # include momentum to speed up learning
        self.w_update = self.momentum * self.w_update + grad_w
        # Move along gradient
        return w - self.learningRate * self.w_update

class Adam():
    # Favorable to other techniques
    def __init__(self, learningRate=0.001, b1=0.9, b2=0.999):
        self.learningRate = learningRate
        self.eps = 1e-8
        # mean/uncentered variance of gradients
        self.m = np.array([])
        self.v = np.array([])
        # Decay rates (suggested by authors of ADAM optimizer)
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_w):
        # Initialize
        if not self.m.any():
            self.m = np.zeros(np.shape(grad_w))
            self.v = np.zeros(np.shape(grad_w))
        # Update mean/variance of gradients w/ exponential decaying average
        self.m = self.b1 * self.m + (1 - self.b1) * grad_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_w, 2)
        # Counteract bias toward 0 (when decay rates are small)
        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)
        # Update rule
        self.w_update = self.learningRate / (np.sqrt(v_hat) + self.eps) * m_hat

        return w - self.w_update
