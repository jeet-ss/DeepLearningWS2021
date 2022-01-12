import numpy as np
import math


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.weight_tensor_update = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.weight_tensor_update is None:
            self.weight_tensor_update = np.zeros(np.shape(weight_tensor))
        self.weight_tensor_update = self.momentum_rate * self.weight_tensor_update - self.learning_rate * gradient_tensor
        if self.regularizer :
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor + self.weight_tensor_update


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.v = 0
        self.r = 0
        self.k = 0
        self.mu = mu
        self.rho = rho
        self.eps = np.finfo(np.dtype(float)).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor

        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))

        weight_tensor_update = self.learning_rate * ( v_hat / (np.sqrt(r_hat) + self.eps))
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - weight_tensor_update
