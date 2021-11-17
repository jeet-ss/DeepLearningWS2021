import copy
import numpy as np
from Layers.Base import BaseLayer

class NeuralNetwork(BaseLayer):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = 0
        self.loss_layer = 0

        self.label_tensor=0
        self.input_tensor=0
        self.error_tensor=0

    def forward(self):
        self.input_tensor , self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in np.flip(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor