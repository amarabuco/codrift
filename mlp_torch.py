import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import optuna

class MLP(nn.Module):
    def __init__(self, input_size, h_size, activation, device=None):
        super(MLP, self).__init__()
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = 1
        self._device = device
        self.activation_name = activation
        
        self.hidden1 = nn.Linear(input_size, h_size)

        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid

    def forward(self, x):
        x = self.hidden1(x)
        return self._activation(x)
        

