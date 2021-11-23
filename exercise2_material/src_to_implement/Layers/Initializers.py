import numpy as np


class Constant:
    def __init__(self , constant_value = 0.1):
        self.constant_val = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant_val)

class UniformRandom:
    def __init__(self):
        self.output = 0

    def initialize(self, weights_shape, fan_in, fan_out):
        self.output= np.random.random(weights_shape)
        return self.output

class Xavier:
    def __init__(self):
        self.output = 0

    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2 / (fan_in + fan_out))
        self.output = np.random.normal(0, std_dev, weights_shape)
        return self.output

class He:
    def __init__(self):
        self.output = 0

    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2 / fan_in)
        self.output = np.random.normal(0, std_dev, weights_shape)
        return self.output