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
        if len(convolution_shape) == 1:
            convolution_shape_1D = np.append(convolution_shape, [1])
            self.convolution_shape = tuple(convolution_shape_1D)

        self.weights_shape = (num_kernels,) + self.convolution_shape
        self.weights = np.random.uniform(0.0, 1.0, self.weights_shape)
        # self.weights = np.random.uniform(0.0,1.0, (np.concatenate(([self.num_kernels], self.convolution_shape))))
        # weight_shape = (num_kernels, self.convolution_shape)
        self.bias = np.random.randn(num_kernels)

        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.random.randn(num_kernels)

        self.cnn_params = {"stride": self.stride_shape, "pad": 1}
        self.cache = None
        # added extra
        self.input_tensor = None
        self.image_size = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #
        image_size = np.shape(input_tensor)[2::]
        self.image_size = image_size
        # output for all images
        feature_map = np.zeros((np.concatenate(((1, self.num_kernels), image_size))))
        # loop through image tensor
        for image in self.input_tensor:
            # output for one image
            # size is of one image dimension without the channels
            features = np.zeros((np.concatenate(((1,), self.image_size))))
            # loop through the kernels
            for i in range(self.num_kernels):
                # w = self.weights[i]
                forward_conv = signal.correlate(image, self.weights[i], mode='same')
                forward_conv1 = np.sum(forward_conv, axis=0) + self.bias[i]
                # stack up the features from each kernel
                features = np.append(features, [forward_conv1], axis=0)
            # stack up features for each image
            feature_map = np.append(feature_map, [features[1::]], axis=0)
        # get rid of the layer with only zeros
        feature_map = feature_map[1::]

        # Strides
        if len(self.stride_shape) == 2:
            feature_map = feature_map[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]]
        elif len(self.stride_shape) == 1:
            feature_map = feature_map[:, :, 0::self.stride_shape[0]]

        return feature_map

    def backward(self, error_tensor):

        # Up-sampling
        error_empty = np.zeros(np.concatenate(((error_tensor.shape[0], error_tensor.shape[1]), self.image_size)))
        if np.ndim(error_tensor) == 4:
            error_empty[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor
        elif np.ndim(error_tensor) == 3:  # 1D case
            error_empty[:, :, 0::self.stride_shape[0]] = error_tensor
        self.error_tensor = error_empty

        # reshaping weights
        back_weights = np.copy(self.weights)
        print("bw", back_weights.shape)
        back_weights = np.swapaxes(back_weights, 0, 1)
        self.weights = back_weights
        print("bw", back_weights.shape)

        # gradient wrt lower layer
        feature_map_back = np.zeros((np.concatenate(((1, self.weights.shape[0]), self.image_size))))
        print(feature_map_back.shape)
        for layer in self.error_tensor:
            features_back = np.zeros((np.concatenate(((1,), self.image_size))))
            for i in range(self.weights.shape[0]):
                print("w", self.weights[i].shape, "l", layer.shape)
                back_conv = signal.convolve(layer, self.weights[i], mode='same')
                back_conv = np.sum(back_conv, axis=0)
                print("b", back_conv.shape, "f", features_back.shape)
                features_back = np.append(features_back, [back_conv], axis=0)
                print("fb", features_back.shape)
            print("fea", feature_map_back.shape)
            feature_map_back = np.append(feature_map_back, [features_back[1::]], axis=0)
        feature_map_back = feature_map_back[1::]

        # gradient wrt weights


        return feature_map_back


        '''
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
        '''

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
