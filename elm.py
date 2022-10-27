import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import optuna

class ELM():
    def __init__(self, input_size, h_size, activation, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = 1
        self._device = device
        self.activation_name = activation

        self._alpha = nn.init.xavier_uniform_(torch.empty(self._input_size, self._h_size, device=self._device))
        self._beta = nn.init.xavier_uniform_(torch.empty(self._h_size, self._output_size, device=self._device))

      

        self._bias = torch.zeros(self._h_size, device=self._device)

        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.nn.relu
        elif activation == 'selu':
            self._activation = torch.nn.selu
        elif activation == 'elu':
            self._activation = torch.nn.elu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)

