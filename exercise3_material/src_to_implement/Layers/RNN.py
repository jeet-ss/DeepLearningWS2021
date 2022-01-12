from Layers.Base import BaseLayer
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected
import numpy as np
import copy


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.weights = None
        # trainable
        self.trainable = True
        # memorize
        self.memorize = False
        #
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # weights and biases
        #self.weights_h = np.random.rand(self.input_size + self.hidden_size + 1, self.output_size)
        #self.bias_h = np.ones(self.output_size)
        # calling fc layer
        self.fc_1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_2 = FullyConnected(self.hidden_size, self.output_size)
        # setting tanh and sigmoid layer
        self.tanH = TanH()
        self.sigmoid = Sigmoid()
        # members for forward
        self.time_instances = 0
        self.input_tensor = None
        self.hidden_state = None
        self.concatenated_x = None
        self.batch_size = 0
        self.y_hat = None
        self.all_input = []
        self.all_hidden = []
        self.all_output = []
        self.fc2_input = []
        # members for backward
        self.error_tensor = None
        self.output_grad = None
        self.input_grad = None
        self.combined_grad = 0
        self.nextlayer_grad = None
        self.hidden_grad = None
        self.weights_grad = None
        self.weights_grad_fc2 = None
        # optimizers
        self._optimizerBias = None
        self._optimizerWeights = None
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        self.time_instances = input_tensor.shape[0]
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        #if self.y_hat is None:
        self.y_hat = np.zeros((self.batch_size, self.output_size))
        # inti
        # initialize hidden state based on batch size
        if self.hidden_state is None or not self.memorize:
            self.hidden_state = np.zeros((self.hidden_size, 1)).T
        # looping through time
        for time, features in enumerate(input_tensor):
            features = features[np.newaxis, :]
            # adding two inputs
            self.concatenated_x = np.hstack((self.hidden_state, features))
            #self.all_input.append(self.concatenated_x)
            # first fc
            self.hidden_state = self.tanH.forward(self.fc_1.forward(self.concatenated_x))
            self.all_hidden.append(self.hidden_state)
            self.all_input.append(self.fc_1.input)
            # 2nd fc
            self.y_hat[time] = self.sigmoid.forward(self.fc_2.forward(self.hidden_state))
            self.fc2_input.append(self.fc_2.input)
            self.all_output.append(self.y_hat[time])

        return self.y_hat

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.nextlayer_grad = np.zeros((error_tensor.shape[0], self.input_size))
        self.weights_grad = np.zeros(self.fc_1.weights.shape)  # remove the bias
        self.weights_grad_fc2 = np.zeros(self.fc_2.weights.shape)
        # new changed
        # should I keep the bias weights from the passes
        bias = np.zeros(self.fc_1.weights.shape)[:1, :]
        bias_2 = np.zeros(self.fc_2.weights.shape)[:1, :]
        # initialize hidden
        self.hidden_grad = np.zeros((1, self.hidden_size))

        for time, error in enumerate(reversed(error_tensor)):
            self.sigmoid.output = self.all_output[-1-time]
            self.fc_2.input = self.fc2_input[-1-time]
            self.tanH.output = self.all_hidden[-1-time]
            self.fc_1.input = self.all_input[-1-time]
            # out of 2nd fc
            self.output_grad = self.fc_2.backward(self.sigmoid.backward(error)[np.newaxis, :])
            self.weights_grad_fc2 += self.fc_2.gradient_weights
            # add self.hidden_state and self.grad_output
            self.combined_grad = np.add(self.output_grad, self.hidden_grad)
            # compute next gradient
            self.input_grad = self.fc_1.backward(self.tanH.backward(self.combined_grad))
            #
            self.weights_grad += self.fc_1.gradient_weights
            # split grad into two parts
            self.hidden_grad = self.input_grad[:, :self.hidden_size]
            self.nextlayer_grad[-1-time] = self.input_grad[:, self.hidden_size:]

        #self.weights_grad = np.concatenate((self.weights_grad, bias), axis=0)
        self.gradient_weights = self.weights_grad

        if self.optimizer:
            # CHANGED the [1:, :] to leave out bias
            #self.fc_2_gradient_weights = np.concatenate((self.fc_2.gradient_weights, bias_2) ,axis=0)
            self.fc_1.weights = self.optimizer.calculate_update(self.fc_1.weights, self.weights_grad)
            self.fc_2.weights = self._optimizer2.calculate_update(self.fc_2.weights, self.weights_grad_fc2)

        return self.nextlayer_grad

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_1.initialize(weights_initializer, bias_initializer)
        self.fc_2.initialize(weights_initializer, bias_initializer)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self._optimizer2 = copy.deepcopy(optimizer)
        self._optimizerWeights = copy.deepcopy(optimizer)
        self._optimizerBias = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, val):
        self._memorize = val

    @property
    def weights(self):
        return self.fc_1.weights

    @weights.setter
    def weights(self, w):
        self.fc_1.weights = w
