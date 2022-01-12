import copy
import numpy as np

class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__()
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.loss = []
        self.layers = []
        self.data_layer = 0
        self.loss_layer = 0

        self.label_tensor=0
        self.input_tensor=0
        self.error_tensor=0
        # phase
        self._phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_temp):
        # loop through the layers and set the testing_phase of each layer with self._phase
        self._phase = phase_temp

    def forward(self):
        self.input_tensor , self.label_tensor = self.data_layer.next()
        regularization_loss = 0
        for layer in self.layers:
            # cannot get any weights
            if layer.trainable == True and layer.optimizer.regularizer is not None:
                regularization_loss += self.optimizer.regularizer.norm(layer.weights)
            self.input_tensor = layer.forward(self.input_tensor)
        layer_loss = self.loss_layer.forward(self.input_tensor, self.label_tensor) + regularization_loss
        return layer_loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in np.flip(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self._phase = False
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self._phase = True
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)

        return input_tensor
