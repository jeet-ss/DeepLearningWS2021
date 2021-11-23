from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor[self.input_tensor <= 0] = 0
        return self.input_tensor

    def backward(self, error_tensor):
        error_tensor_previous_layer = error_tensor * (self.input_tensor > 0)
        return error_tensor_previous_layer
