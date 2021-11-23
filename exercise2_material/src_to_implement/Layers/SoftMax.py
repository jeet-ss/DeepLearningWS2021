from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_tensor = 0
        self.y_hat = 0
        self.error_tensor = 0
        self.out = 0

    def forward(self, input_tensor):
        # set the data
        self.input_tensor = input_tensor
        # find exponent for each data point
        exp_x = np.exp(self.input_tensor-np.max(input_tensor, axis=1).reshape(-1, 1))
        # sum along exp of the entire batch
        sum_exp_x = np.sum(exp_x, axis=1).reshape(-1, 1)
        # normalize the exp between 0 and 1
        normalized_exp_x = exp_x / sum_exp_x
        # output
        self.y_hat = normalized_exp_x
        return normalized_exp_x

    def backward(self, error_tensor):
        # set
        self.error_tensor = error_tensor
        # calculate update term
        update = np.sum(error_tensor * self.y_hat, axis=1).reshape(-1, 1)
        # update error value
        self.error_tensor -= update
        # prepare for next layer
        self.out = self.error_tensor * self.y_hat
        return self.out
