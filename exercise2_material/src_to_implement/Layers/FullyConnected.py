import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # initial random weights
        weights = np.random.rand(self.input_size + 1, self.output_size)
        self.weights = weights
        self._optimizer = None
        self.input = 0

    def forward(self, input_tensor):
        self.input = np.c_[input_tensor, np.ones((input_tensor.shape[0], 1))]
        self.weights = self.weights
        # Y_hat = np.dot(X, W)
        return np.dot(self.input, self.weights)

    def backward(self, error_tensor):
        x = self.input
        errorpre = np.dot(
            error_tensor,
            self.weights[0:self.weights.shape[0]-1, :].T
        )
        self.gradient_tensor = np.dot(x.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights,
                self.gradient_tensor
            )
        return errorpre

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.weights.shape[0]-1, self.weights.shape[1]), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.weights.shape[1]), self.input_size, self.output_size)
        self.weights = np.vstack((weights, bias))