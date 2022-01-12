import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self.bias = np.ones(self.output_size)
        self.input = 0
        self.gradient_tensor = 0
        self._optimizer = None

    def forward(self, input_tensor):
        X = np.c_[input_tensor, np.ones((input_tensor.shape[0], 1))]
        W = self.weights
        self.input = X
        Y_hat = np.dot(X, W)
        return Y_hat

    def backward(self, error_tensor):
        error_tensor_previous_layer = np.dot(
            error_tensor,
            self.weights[:, :].T
        )
        self.gradient_tensor = np.dot(self.input.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights,
                self.gradient_tensor
            )
        error_tensor_previous_layer = np.delete(error_tensor_previous_layer, error_tensor_previous_layer.shape[1] - 1, axis=1)
        return error_tensor_previous_layer

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.weights.shape[0]-1, self.weights.shape[1]), self.input_size, self.output_size)
        bias = bias_initializer.initialize(self.bias.shape, self.output_size, 1)
        self.weights = np.vstack((weights, bias))


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
