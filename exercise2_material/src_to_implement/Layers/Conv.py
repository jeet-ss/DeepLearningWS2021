import numpy as np


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape #single val or tuple
        self.convolution_shape = convolution_shape #decides 1D or 2D convolution
        self.num_kernels = num_kernels #integer val

