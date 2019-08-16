import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import hashlib
import json

class FFN(nn.Module):
    def __init__(self, inputs, n_hidden_layers = 2, decay_per_layer = 0.75):
        super(FFN, self).__init__()
        n_inputs = sum(list(inputs.values()))

        hash_object = hashlib.md5(json.dumps(inputs).encode())
        self.name = hash_object.hexdigest()
        #print(self.name)

        self.hidden_layers = nn.ModuleList([nn.Linear(ceil(n_inputs * decay_per_layer ** i), ceil(n_inputs * decay_per_layer ** (i + 1))) for i in range(n_hidden_layers)])

        self.output_layer = nn.Linear(ceil(n_inputs * decay_per_layer ** n_hidden_layers), 3)

    # TODO: have different outputs depending on the type of training
    def forward(self, x):
        hidden = x
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = F.relu(hidden)
        output = self.output_layer(hidden)
        output = F.softmax(output, dim=1)
        return output
