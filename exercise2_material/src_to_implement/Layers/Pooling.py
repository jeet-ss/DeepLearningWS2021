import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.input_tensor = 0

    def forward(self, input_tensor):
        # returns input tensor for next layer
        self.input_tensor = input_tensor

    def backward(self, error_tensor):
        pass
