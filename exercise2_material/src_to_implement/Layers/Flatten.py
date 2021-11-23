import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        #
        batch, x,y,s = input_tensor.shape
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(batch,  x * y * s)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
