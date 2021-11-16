import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        super().__init__()

    def forward(self, prediction_tensor, label_tensor):
        self.prediction = prediction_tensor
        self.label = label_tensor
        #
        #for idx, item in np.ndenumerate(self.prediction):
        #    if self.label[idx] == 1:
         #       ce_Loss += -np.log(item + np.finfo(np.dtype(float)).eps)
        #
        ce_Loss = np.sum(- np.log(self.prediction[self.label == 1] + np.finfo(np.dtype(float)).eps))
        #
        self.loss = ce_Loss
        return self.loss


    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        #
        error = -(self.label_tensor/self.prediction)
        return error