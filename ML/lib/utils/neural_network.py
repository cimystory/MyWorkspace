import math
import numpy as np
from data_manipulation import one_hot_encode, shuffle_data, train_test_split
import progressbar
bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

# Neural Net Superclass


class NeuralNet():
    """Generic class to model NeuralNet
    """

    def __init__(self, optimizer, loss, validation_data=None):
        # initialize Neural Net superclass
        self.loss = loss
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.X_val = np.empty([])
        self.y_val = np.empty([])
        # populate validation data
        if validation_data:
            self.X_val, y_val = validation_data
            self.y_val = one_hot_encode(y_val)

    def addLayer(self, layer):
        # if there are layers int he network already then,
        #     set new layers input shape as previous layers output shape
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        # set optimizer
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        # add layer to list
        self.layers.append(layer)

    def train(self, X, y, n_epochs, batch_size):

        # Convert to one-hot encoding
        y = one_hot_encode(y)  # binarize categorical data

        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)
        # To visualize progress
        bar = progressbar.ProgressBar(widget=bar_widgets)
        for _ in bar(range(n_epochs)):
            # Shuflle through batches
            idx = np.arange(n_samples)
            np.random.shuffle(idx)

            batch_error = 0   # Mean batch training error
            for i in range(n_batches):
                X_batch = X[idx[i * batch_size:(i + 1) * batch_size]]
                y_batch = y[idx[i * batch_size:(i + 1) * batch_size]]
                loss = self.batch_train(X_batch, y_batch)
                batch_error += loss

            # Save the epoch mean error
            self.errors["training"].append(batch_error / n_batches)
            if self.X_val.any():
                # Calculate the validation error
                y_val_hat = self._forward_pass(self.X_val)
                validation_loss = np.mean(self.loss.function(self.y_val, y_val_hat))
                self.errors["validation"].append(validation_loss)

        return self.errors["training"], self.errors["validation"]

    def batch_train(self, X, y):
        # Calculate output
        y_hat = self._forward_pass(X)
        # Calculate the training loss
        loss = np.mean(self.loss.function(y, y_hat))
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss.gradient(y, y_hat)
        # Backpropagate to Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss

    def set_trainingState(self, trainingState=True):
        for layer in self.layers:
            layer.trainingState = trainingState

    def _forward_pass(self, X, trainingState=True):
        # Calculate the output of the NN. The output of layer l1 becomes the
        # input of the following layer l2
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, trainingState)

        return layer_output

    def _backward_pass(self, loss_grad):
        # Propogate the gradient 'backwards' and update the weights for each layer
        prop_grad = loss_grad
        for layer in reversed(self.layers):
            prop_grad = layer.backward_pass(prop_grad)

    # Use the trained model to predict labels of X
    def predict(self, X):
        return self._forward_pass(X, trainingState=False)

    # Use the trained model to hallucinate X
    def backQuery(self, y):
        layer_output = y
        self.set_trainingState(trainingState=False)

        for layer in reversed(self.layers):
            layer_output = layer.backwardQuery(layer_output)
        return layer_output
    #
