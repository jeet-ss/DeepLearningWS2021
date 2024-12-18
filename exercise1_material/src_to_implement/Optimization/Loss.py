import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        super().__init__()

    def forward(self, prediction_tensor, label_tensor):
        self.prediction = prediction_tensor
        self.label = label_tensor
        #
        loss = np.sum(- np.log(self.prediction[self.label == 1] + np.finfo(np.dtype(float)).eps))
        #
        self.loss = loss
        return self.loss


    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        #
        error = -(self.label_tensor/(self.prediction + np.finfo(np.dtype(float)).eps))
        return error