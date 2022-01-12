from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import numpy as np
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.alpha = 0.8 #check what to init

        # initialize weights and bias
        #weight-gamma, bias - beta
        self.weights = 0
        self.bias = 0
        self.weights_initializier = None
        self.bias_initializer = None
        self.initialize(self.weights_initializier, self.bias_initializer)
        # optimizers
        self._optimizerBias = None
        self._optimizerWeights = None
        self._optimizer = None
        self._gradient_bias = 0
        self._gradient_weights = 0
        # random inits
        self.input_tensor = None
        self.input_shape = 0
        self.output = None
        self.batch_size = None
        self.batch_mean = 0
        self.batch_sigma = 0
        self.mean_ma = None
        self.sigma_ma = None
        self.prev_mean = 0
        self.prev_sigma = 0
        self.normalized_input = 0
        self.error_tensor = 0
        self.input_grad = 0

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def reformat(self, tensor):
        if np.ndim(tensor) == 4:
            b,h,m,n = tensor.shape
            tensor_temp = np.reshape(tensor, (b, h, m*n))
            tensor_temp = np.swapaxes(tensor_temp, 1, 2)
            tensor_out = np.reshape(tensor_temp, (b*m*n, h))
        elif np.ndim(tensor) == 2:
            b,h,m,n = self.input_shape
            tensor_temp = np.reshape(tensor, (b, m*n, h))
            tensor_temp = np.swapaxes(tensor_temp, 1, 2)
            tensor_out = np.reshape(tensor_temp, (b, h, m, n))
        else:
            tensor_out = tensor
        return tensor_out

    def forward(self,input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = input_tensor

        #check for 4D case
        if np.ndim(input_tensor) == 4 :
            self.input_tensor = self.reformat(input_tensor)

        # calculate mean along the batch
        self.batch_mean = np.mean(self.input_tensor, axis=0)
        self.batch_sigma = np.std(self.input_tensor, axis=0)
        #
        if self.mean_ma is None and self.sigma_ma is None:
            self.mean_ma = self.batch_mean
            self.sigma_ma = self.batch_sigma

        # calculate moving average for test phase
        if self.testing_phase == False:
            self.mean_ma = self.alpha * self.prev_mean + (1 - self.alpha) * self.batch_mean
            self.sigma_ma = self.alpha * self.prev_sigma + (1 - self.alpha) * self.batch_sigma

        self.prev_mean = self.mean_ma
        self.prev_sigma = self.sigma_ma

        if self.testing_phase == True:
            self.batch_mean = self.mean_ma
            self.batch_sigma = self.sigma_ma

        self.normalized_input = (self.input_tensor - self.batch_mean[np.newaxis, :])/(np.sqrt(self.batch_sigma[np.newaxis, :]**2 + np.finfo(float).eps))
        self.output = self.normalized_input * self.weights + self.bias

        if np.ndim(input_tensor) == 4 :
            self.output = self.reformat(self.output)

        return self.output

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        #self.input_shape = error_tensor.shape

        if np.ndim(error_tensor) == 4 :
            self.error_tensor = self.reformat(error_tensor)

        # compute grad wrt everything
        self._gradient_weights = np.sum(self.error_tensor * self.normalized_input, axis=0, keepdims=True) # sum along the batch
        self._gradient_bias = np.sum(self.error_tensor, axis=0, keepdims=True)
        self.input_grad = compute_bn_gradients(self.error_tensor, self.input_tensor, self.weights, self.batch_mean, self.batch_sigma**2)

        # update weights and bias
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizerBias.calculate_update(self.bias, self._gradient_bias)

        if np.ndim(error_tensor) == 4 :
            self.input_grad = self.reformat(self.input_grad)

        return self.input_grad

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizerWeights = copy.deepcopy(optimizer)
        self._optimizerBias = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
