import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import hashlib
import json

class FFN(nn.Module):
    def __init__(self, inputs, batch_size, n_hidden_layers = 4, decay_per_layer = 0.85):
        super(FFN, self).__init__()
        self.n_inputs = sum(list(inputs.values()))# - 4
        self.batch_size = batch_size

        hash_object = hashlib.md5(json.dumps(inputs).encode())
        self.name = hash_object.hexdigest()
        #print(self.name)

        self.dropout = nn.Dropout(0.25)

        self.lstm = nn.LSTM(self.n_inputs, self.n_inputs)
        self.state = (torch.randn(1, self.batch_size, self.n_inputs), torch.randn(1, self.batch_size, self.n_inputs))

        self.hidden_layers = nn.ModuleList([nn.Linear(ceil(self.n_inputs * decay_per_layer ** i), ceil(self.n_inputs * decay_per_layer ** (i + 1))) for i in range(n_hidden_layers)])

        self.output_layer = nn.Linear(ceil(self.n_inputs * decay_per_layer ** n_hidden_layers), 3)

    def init_state(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        self.state = (torch.randn(1, batch_size, self.n_inputs), torch.randn(1, batch_size, self.n_inputs))

    # TODO: have different outputs depending on the type of training
    def forward(self, x):
        hidden, self.state = self.lstm(x, self.state)
        for layer in self.hidden_layers:
            hidden = self.dropout(layer(hidden))
            hidden = F.leaky_relu(hidden)
        output = self.output_layer(hidden)
        output = F.softmax(output, dim=-1)
        return output
