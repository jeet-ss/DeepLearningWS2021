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
        pass

    def train(self, iterations):
        pass

    def test(self, input_tensor):
        pass