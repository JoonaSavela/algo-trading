import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import hashlib
import json
from collections import namedtuple, deque
import random

class FFN(nn.Module):
    def __init__(self, inputs, batch_size, use_lstm = True, Qlearn = False, n_hidden_layers = 4, decay_per_layer = 0.85, hidden_size = None):
        super(FFN, self).__init__()
        self.n_inputs = sum(list(inputs.values()))
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.Qlearn = Qlearn

        if hidden_size is None:
            hidden_size = self.n_inputs

        self.hidden_size = hidden_size

        hash_object = hashlib.md5(json.dumps(inputs).encode())
        self.name = hash_object.hexdigest()
        #print(self.name)

        self.dropout = nn.Dropout(0.25)

        if self.use_lstm:
            self.lstm = nn.LSTM(self.n_inputs, self.hidden_size)
            self.state = (torch.randn(1, self.batch_size, self.hidden_size), torch.randn(1, self.batch_size, self.hidden_size))

        self.hidden_layers = nn.ModuleList([nn.Linear(ceil(self.hidden_size * decay_per_layer ** i), ceil(self.hidden_size * decay_per_layer ** (i + 1))) for i in range(n_hidden_layers)])

        self.output_layer = nn.Linear(ceil(self.hidden_size * decay_per_layer ** n_hidden_layers), 3)

    def init_state(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        self.state = (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, x):
        if self.use_lstm:
            hidden, self.state = self.lstm(x, self.state)
        else:
            hidden = x

        for layer in self.hidden_layers:
            hidden = self.dropout(layer(hidden))
            hidden = F.leaky_relu(hidden)

        output = self.output_layer(hidden)
        if not self.Qlearn:
            output = F.softmax(output, dim=-1)

        return output


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
