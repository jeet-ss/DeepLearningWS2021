import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.pooling_shape = pooling_shape
        self.stride_shape = stride_shape
        self.cache = {}
        self.p = 0

    def forward(self, input_tensor):
        # Preparing variables with appropriate shapes
        N, F, H, W = input_tensor.shape  # For input

        self.cache = input_tensor

        # Preparing variables with parameters
        pooling_height = self.pooling_shape[0]
        pooling_width = self.pooling_shape[1]


        # Defining spatial size of output image volume after pooling layer
        height_pooled_out = int(1 + (H - pooling_height) / self.stride_shape[0])
        width_polled_out = int(1 + (W - pooling_width) / self.stride_shape[1])

        # Creating zero valued volume for output image volume after pooling layer
        pooled_output = np.zeros((N, F, height_pooled_out, width_polled_out))

        # Implementing forward naive pooling pass through N input images,
        # every with F channels (number of feature maps)
        # And calculating output pooled image volume
        # For every image
        for n in range(N):
            # Going through all input image through all channels
            for i in range(height_pooled_out):
                for j in range(width_polled_out):
                    # Preparing height and width for current pooling region
                    ii = i * self.stride_shape[0]
                    jj = j * self.stride_shape[1]
                    # Getting current pooling region with all channels F
                    current_pooling_region = input_tensor[n, :, ii:ii+pooling_height, jj:jj+pooling_width]
                    # Finding maximum value for all channels and filling output pooled image
                    # Reshaping current pooling region from (3, 2, 2) - 3 channels and 2 by 2
                    # To (3, 4) in order to utilize np.max function
                    # Specifying 'axis=1' as parameter for choosing maximum value
                    # out of 4 numbers along 3 channels
                    pooled_output[n, :, i, j] = \
                        np.max(current_pooling_region.reshape((F, pooling_height * pooling_width)), axis=1)

        # Returning output resulted data
        return pooled_output

    def backward(self, error_tensor):
        # Preparing variables with appropriate shapes
        x = self.cache
        N, F, H, W = x.shape

        # Preparing variables with parameters
        pooling_height = self.pooling_shape[0]
        pooling_width = self.pooling_shape[1]

        # Defining spatial size of output image volume after pooling layer
        height_pooled_out = int(1 + (H - pooling_height) / self.stride_shape[0])
        width_polled_out = int(1 + (W - pooling_width) / self.stride_shape[1])
        # Depth of output volume is number of channels which is F (or number of feature maps)
        # And number of input images N remains the same - it is number of output image volumes now
        # Creating zero valued volume for output gradient after backward pass of pooling layer
        # The shape is the same with x.shape
        dx = np.zeros((N, F, H, W))

        # Implementing backward naive pooling pass through N input images,
        # every with F channels (number of feature maps)
        # And calculating output pooled image volume
        # For every image
        for n in range(N):
            # For every channel
            for f in range(F):
                # Going through all pooled image by height and width
                for i in range(height_pooled_out):
                    for j in range(width_polled_out):
                        # Preparing height and width for current pooling region
                        ii = i * self.stride_shape[0]
                        jj = j * self.stride_shape[1]
                        # Getting current pooling region
                        current_pooling_region = x[n, f, ii:ii+pooling_height, jj:jj+pooling_width]
                        # Finding maximum value for current pooling region
                        current_maximum = np.max(current_pooling_region)
                        # Creating array with the same shape as 'current_pooling_region'
                        # Filling with 'True' and 'False' according to the condition '==' to 'current_maximum'
                        temp = current_pooling_region == current_maximum
                        # Calculating output gradient
                        dx[n, f, ii:ii+pooling_height, jj:jj+pooling_width] += \
                            error_tensor[n, f, i, j] * temp

        return dx
