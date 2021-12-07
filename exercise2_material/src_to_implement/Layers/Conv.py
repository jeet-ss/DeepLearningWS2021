from scipy import signal
import numpy as np
import copy
import math
from Layers.Base import BaseLayer
import math


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape  # single val or tuple
        self.num_kernels = num_kernels  # integer val
        self.convolution_shape = convolution_shape  # decides 1D or 2D convolution

        # self.weights = np.linspace(-1.0, 1.0, num=np.prod(weight_shape), dtype='float64').reshape(weight_shape)
        self.weights = np.random.uniform(0,1, (np.concatenate(([self.num_kernels], self.convolution_shape))))
        # weight_shape = (num_kernels, self.convolution_shape)
        self.bias = np.random.randn(num_kernels)

        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.random.randn(num_kernels)

        self.cnn_params = {"stride": self.stride_shape, "pad": 1}
        self.cache = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        image_size  = np.shape(input_tensor)[2::]
        # output for all images
        feature_map = np.zeros((np.concatenate(((1, self.num_kernels), image_size))))
        # loop through image tensor
        for image in self.input_tensor:
            # output for one image
            # size is of one image dimension without the channels
            features = np.zeros((np.concatenate(((1,), image_size))))
            # loop through the kernels
            for i in range(self.num_kernels):
                w = self.weights[i]
                forward_conv = signal.correlate(image, self.weights[i], mode='same')

                forward_conv1 = np.sum(forward_conv, axis=0) + self.bias[i]
                # stack up the features from each kernel
                features = np.append(features, [forward_conv1], axis = 0)
            # stack up features for each image
            feature_map = np.append(feature_map, [features[1::]], axis = 0)
        # get rid of the layer with only zeros
        feature_map = feature_map[1::]

        # Strides
        if len(self.stride_shape) == 2:
            feature_map = feature_map[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]]

        return feature_map



    def backward(self, error_tensor):
        derivative_out = error_tensor
        x, w, b, cnn_params = self.cache

        N, C, H, W = x.shape  # For input
        F, _, HH, WW = w.shape  # For weights
        _, _, height_out, weight_out = derivative_out.shape  # For output feature maps

        stride = cnn_params['stride']
        pad = cnn_params['pad']

        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
        dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for n in range(N):
            for f in range(F):
                for i in range(0, H, stride[0]):
                    for j in range(0, W, stride[1]):
                        dx_padded[n, :, i:i + HH, j:j + WW] += w[f, :, :, :] * derivative_out[n, f, i, j]
                        dw[f, :, :, :] += x_padded[n, :, i:i + HH, j:j + WW] * derivative_out[n, f, i, j]
                        db[f] += derivative_out[n, f, i, j]

        dx = dx_padded[:, :, 1:-1, 1:-1]
        return dx

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,
                                                      np.prod(self.convolution_shape),
                                                      self.num_kernels * self.convolution_shape[0] *
                                                      self.convolution_shape[1])
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                np.prod(self.convolution_shape),
                                                np.prod(self.convolution_shape))

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.optimizerWeigths = copy.deepcopy(self.__optimizer)
        self.optimizerBias = copy.deepcopy(self.__optimizer)

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.__gradient_bias = gradient_bias

