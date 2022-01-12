from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase is False:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability
            input_tensor = input_tensor * self.mask
        return input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.mask
