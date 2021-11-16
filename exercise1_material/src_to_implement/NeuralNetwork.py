import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = 0
        self.loss_layer = 0


    def forward(self):
        pass

    def backward(self):
        pass

    def append_layer(self, layer):
        optimizer_copy = copy.deepcopy(self.optimizer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)

    def test(self, input_tensor):
        pass