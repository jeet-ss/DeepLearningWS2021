import numpy as np
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape  # single val or tuple
        self.convolution_shape = convolution_shape  # decides 1D or 2D convolution
        self.num_kernels = num_kernels  # integer val

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass
