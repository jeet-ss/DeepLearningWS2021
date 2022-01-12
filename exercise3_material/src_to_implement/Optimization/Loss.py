import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        super().__init__()

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        # np.finfo(np.dtype(float)).eps))
        # smallest possible positive number that the float datatype can represent on the machine
        loss = np.sum(-np.log(self.prediction_tensor[label_tensor == 1] + np.finfo(np.dtype(float)).eps))
        return loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        error_tensor = -(self.label_tensor/(self.prediction_tensor + np.finfo(np.dtype(float)).eps))
        return error_tensor
