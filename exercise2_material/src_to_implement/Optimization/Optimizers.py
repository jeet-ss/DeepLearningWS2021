import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.value = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.value = (self.momentum_rate * self.value) - (self.learning_rate * gradient_tensor)
        weight_tensor = weight_tensor + self.value
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.v = 0
        self.g = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.g = gradient_tensor
        self.v = (self.mu * self.v) + ((1-self.mu) * self.g)
        self.r = (self.rho * self.r) + ((1-self.rho) * self.g * self.g)  # not sure if circular multiplication will work
        self.k += 1
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        updated_gradient = v_hat / (np.sqrt(r_hat) + np.finfo(np.dtype(float)).eps)
        updated_weight = weight_tensor - (self.learning_rate * updated_gradient)
        return updated_weight
