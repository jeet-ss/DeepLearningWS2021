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
        self.back_weights = None

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
        print("forwrad out = ", feature_map.shape)
        return feature_map

    def backward(self, error_tensor):
        print("error_tensor", error_tensor.shape)
        # Up-sampling
        error_empty = np.zeros(np.concatenate(((error_tensor.shape[0], error_tensor.shape[1]), self.image_size)))
        if np.ndim(error_tensor) == 4:
            error_empty[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor
        elif np.ndim(error_tensor) == 3:  # 1D case
            error_empty[:, :, 0::self.stride_shape[0]] = error_tensor
        self.error_tensor = error_empty
        print("input", self.input_tensor.shape)
        print("error", self.error_tensor.shape)
        # reshaping weights
        # print("fw", self.weights.shape)
        back_weights = np.copy(self.weights)
        back_weights = np.swapaxes(back_weights, 0, 1)  # swap kernel no and channel
        back_weights = np.flip(back_weights, 1)  # flip along channels
        self.back_weights = back_weights
        # print("bw" , self.back_weights.shape)

        # gradient wrt lower layer
        feature_map_back = np.zeros((np.concatenate(((1, self.back_weights.shape[0]), self.image_size))))
        for layer in self.error_tensor:
            # print("la", layer.shape)
            features_back = np.zeros((np.concatenate(((1,), self.image_size))))
            for i in range(self.back_weights.shape[0]):
                back_conv = signal.convolve(layer, self.back_weights[i], mode='same')
                back_conv = np.sum(back_conv, axis=0)
                features_back = np.append(features_back, [back_conv], axis=0)
            feature_map_back = np.append(feature_map_back, [features_back[1::]], axis=0)
        feature_map_back = feature_map_back[1::]
        print(feature_map_back.shape)

        # gradient wrt weights
        back_features = np.copy(feature_map_back)
        # print("before pad", back_features.shape)
        # padding calculation
        if np.ndim(self.weights) == 4:
            pad_y = ((self.weights.shape[2] - 1) / 2)
            pad_x = ((self.weights.shape[3] - 1) / 2)
            print(pad_x, pad_y)
            # check for y
            if float.is_integer(pad_y):
                pad_y1 = math.floor(pad_y)
                pad_y2 = math.floor(pad_y)
            else:
                pad_y1 = math.floor(pad_y)
                pad_y2 = math.floor(pad_y) + 1
            # check for x
            if float.is_integer(pad_x):
                pad_x1 = math.floor(pad_x)
                pad_x2 = math.floor(pad_x)
            else:
                pad_x1 = math.floor(pad_x)
                pad_x2 = math.floor(pad_x) + 1
            print(pad_x1, pad_x2, pad_y1, pad_y2)
            padded_image = np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant',
                                  constant_values=0.0)
            print("after pad", padded_image.shape)
        elif np.ndim(self.weights) == 3:
            padded_image = self.input_tensor
        # correlate part
        self.gradient_weights = np.zeros(self.weights.shape)
        print("grad", self.gradient_weights.shape)
        if np.ndim(self.weights) == 4:
            # bias gradient
            self.gradient_bias = np.sum(np.sum(np.sum(self.error_tensor, axis=3), axis=2), axis=0)
            # weights gradient
            gradient_weights = np.copy(self.gradient_weights)
            for i in range(self.input_tensor.shape[0]):  # 2
                for j in range(self.error_tensor.shape[1]):  # 4
                    for k in range(self.input_tensor.shape[1]):  # 3
                        print(padded_image[i, k, :, :].shape, self.error_tensor[i, j, :, :].shape)
                        # left working here
                        # check that the weights matrix init for the back pass is not
                        # matching and should be realted with something else
                        gradient_weights[j, k, :, :] += signal.correlate(padded_image[i, k, :, :],
                                                                         self.error_tensor[i, j, :, :], mode='valid')

            self.gradient_weights = gradient_weights
            print("res", self.gradient_weights.shape)
        elif np.ndim(self.weights) == 3:
            # bias
            self.gradient_bias = np.sum(np.sum(self.error_tensor, axis=2), axis=0)
            # weights
            gradient_weights = np.copy(self.gradient_weights)
            for i in range(self.input_tensor.shape[0]):
                for j in range(self.num_kernels):
                    for k in range(self.gradient_weights.shape[1]):
                        gradient_weights[j, k, :] += signal.correlate(padded_image[i, k, :],
                                                                         self.error_tensor[i, j, :], mode='valid')
            self.gradient_weights = gradient_weights

        if self.optimizer:
            self.weights = self.optimizerWeigths.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizerBias.calculate_update(self.bias, self.gradient_bias)

        return feature_map_back

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,
                                                      np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1::]))
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                np.prod(self.convolution_shape),
                                                np.prod(self.convolution_shape))

    @property
    def optimizer(self):
        return self._optimizerWeigths is not  None and self._optimizerWeigths is not None

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizerWeigths = copy.deepcopy(optimizer)
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
