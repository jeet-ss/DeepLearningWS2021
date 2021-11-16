import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        weights = np.random.rand(self.input_size + 1, self.output_size)
        self.weights = weights
        self._optimizer = None

    def forward(self, input_tensor):
        X = np.c_[input_tensor, np.ones((input_tensor.shape[0], 1))]
        W = self.weights
        self.input = X
        self.weights = W
        Y_hat = np.dot(X, W)
        return Y_hat

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
