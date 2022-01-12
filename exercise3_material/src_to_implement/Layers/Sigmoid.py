from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = 0

    def forward(self, input_tensor):
        self.output = 1 / (1 + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        derivative = self.output*(1 - self.output)
        return derivative*error_tensor
