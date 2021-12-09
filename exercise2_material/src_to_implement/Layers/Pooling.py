import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.pooling_shape = pooling_shape
        self.stride_shape = stride_shape
        self.pooling_height = self.pooling_shape[0]
        self.pooling_width = self.pooling_shape[1]

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch, self.kernel, self.height, self.width = input_tensor.shape
        self.height_pooled_out = int(1 + (self.height - self.pooling_height) / self.stride_shape[0])
        self.width_polled_out = int(1 + (self.width - self.pooling_width) / self.stride_shape[1])
        output = np.zeros((self.batch, self.kernel, self.height_pooled_out, self.width_polled_out))

        for n in range(self.batch):
            for i in range(self.height_pooled_out):
                for j in range(self.width_polled_out):
                    ii = i * self.stride_shape[0]
                    jj = j * self.stride_shape[1]
                    current_pooling_region = input_tensor[n, :, ii:ii + self.pooling_height, jj:jj + self.pooling_width]
                    output[n, :, i, j] = np.max(current_pooling_region.reshape((self.kernel, self.pooling_height * self.pooling_width)), axis=1)
        return output

    def backward(self, error_tensor):
        error_tensor_next_layer = np.zeros((self.batch, self.kernel, self.height, self.width))
        for n in range(self.batch):
            for f in range(self.kernel):
                for i in range(self.height_pooled_out):
                    for j in range(self.width_polled_out):
                        ii = i * self.stride_shape[0]
                        jj = j * self.stride_shape[1]
                        current_pooling_region = self.input_tensor[n, f, ii:ii + self.pooling_height, jj:jj + self.pooling_width]
                        current_maximum = np.max(current_pooling_region)
                        error_tensor_next_layer[n, f, ii:ii + self.pooling_height, jj:jj + self.pooling_width] += error_tensor[n, f, i, j] * (current_pooling_region == current_maximum)
        return error_tensor_next_layer
