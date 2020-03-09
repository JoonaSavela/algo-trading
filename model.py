import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import hashlib
import json
from collections import namedtuple, deque
import random

class CNN(nn.Module):
    def __init__(
            self,
            n_features,
            n_hidden_features_per_layer,
            kernel_size = 2,
            n_conv_layers = 7):
        super(CNN, self).__init__()
        self.n_features = n_features
        self.n_hidden_features_per_layer = n_hidden_features_per_layer
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.sequence_length = self.kernel_size ** self.n_conv_layers

        # assert(self.sequence_length <= 2000)

        self.conv_layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                self.n_features + i * self.n_hidden_features_per_layer,
                self.n_features + (i + 1) * self.n_hidden_features_per_layer,
                self.kernel_size,
                dilation = self.kernel_size ** i
                # stride = self.kernel_size
            ))
            for i in range(self.n_conv_layers)
        ])

        self.linear1 = nn.utils.weight_norm(nn.Linear(
            self.n_features + self.n_conv_layers * self.n_hidden_features_per_layer,
            (self.n_features + self.n_conv_layers * self.n_hidden_features_per_layer) // 2
        ))

        self.linear2 = nn.utils.weight_norm(nn.Linear(
            (self.n_features + self.n_conv_layers * self.n_hidden_features_per_layer) // 2,
            2 # (keeps, buys)
        ))

    def forward(self, x):
        hidden = x

        # convert input into shape: (batch_size, n_features, sequence_length)
        if hidden.dim() == 2:
            if hidden.shape[1] == self.n_features:
                hidden = hidden.t()
            hidden = hidden.unsqueeze(0)

        for conv in self.conv_layers:
            hidden = conv(hidden)
            hidden = F.leaky_relu(hidden)
            # print(hidden.shape)

        hidden = hidden.permute(0, 2, 1).squeeze(0)
        # print(hidden.shape)

        hidden = self.linear1(hidden)
        hidden = F.leaky_relu(hidden)

        hidden = self.linear2(hidden)

        out = torch.sigmoid(hidden)

        return out # (pred_seq_length, 2); 2 = (keeps, buys)


class FFN(nn.Module):
    def __init__(self, inputs, batch_size, use_lstm = True, Qlearn = False, use_tanh = True, n_hidden_layers = 4, decay_per_layer = 0.85, hidden_size = None, use_behavioral_cloning = True, n_ahead = 30, n_slots = 10):
        super(FFN, self).__init__()
        self.n_inputs = sum(list(inputs.values()))
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.Qlearn = Qlearn
        self.use_tanh = use_tanh
        self.use_behavioral_cloning = use_behavioral_cloning
        self.n_ahead = n_ahead
        self.n_slots = n_slots

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

        if self.use_behavioral_cloning:
            self.output_layer = nn.Linear(ceil(self.hidden_size * decay_per_layer ** n_hidden_layers), 1 if self.use_tanh and not self.Qlearn else 3)
        else:
            self.output_layer = nn.Linear(ceil(self.hidden_size * decay_per_layer ** n_hidden_layers), self.n_ahead * self.n_slots)

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
        if self.use_behavioral_cloning:
            if not self.Qlearn:
                if self.use_tanh:
                    output = torch.tanh(output)
                else:
                    output = F.softmax(output, dim = -1)
        else:
            output = output.view(x.shape[0], x.shape[1], self.n_ahead, self.n_slots)
            output = F.softmax(output, dim = -1)
            # output = torch.sigmoid(output)

        return output


def get_expected_min_max_profits(out, min_max_log_returns, use_max = True):
    n_slots = out.size(-1)
    n_ahead = out.size(-2)
    batch_size = out.size(1)
    sequence_length = out.size(0)

    min_log_rs = min_max_log_returns[:, 0]
    max_log_rs = min_max_log_returns[:, 1]

    block_sizes = (max_log_rs - min_log_rs) / n_slots

    log_returns = (torch.arange(n_slots) + 0.5).float().unsqueeze(0) * block_sizes.unsqueeze(1)
    log_returns += min_log_rs.unsqueeze(1)
    # log_returns = log_returns.float()

    if use_max:
        out = F.one_hot(out.max(-1)[1], n_slots).float()

    expected_profits = torch.exp((out * log_returns).sum(dim = -1))
    expected_profits = torch.cat([torch.ones(sequence_length, batch_size, 1), expected_profits], dim = -1)

    expected_min_profit, expected_min_profit_idx = expected_profits.min(dim = -1)
    expected_max_profit, expected_max_profit_idx = expected_profits.max(dim = -1)

    # for seq_i in range(sequence_length):
    #     for b in range(batch_size):
    #         expected_profits = torch.exp((out[seq_i, b, :, :] * log_returns).sum(dim = -1))

    return expected_profits, expected_min_profit, expected_min_profit_idx, expected_max_profit, expected_max_profit_idx

def get_buys_from_expected_min_max_profits(expected_min_profit, expected_min_profit_idx, expected_max_profit, expected_max_profit_idx):
    sequence_length = expected_min_profit.size(0)

    if len(expected_min_profit_idx.shape) > 1:
        expected_min_profit_idx = expected_min_profit_idx[:, 0]
        expected_max_profit_idx = expected_max_profit_idx[:, 0]

    li = expected_min_profit_idx < expected_max_profit_idx

    buy_li = (expected_min_profit_idx == 0) | (~li & (expected_max_profit_idx > 0))
    sell_li = (expected_max_profit_idx == 0) | (li & (expected_min_profit_idx > 0))

    idx = torch.arange(sequence_length)

    buys = idx[buy_li]
    sells = idx[sell_li]

    buys = buys.numpy()
    sells = sells.numpy()
    buy_li = buy_li.float().numpy()
    sell_li = sell_li.float().numpy()

    return buys, sells, buy_li, sell_li

    # buys_optim = torch.zeros(sequence_length)
    #
    # for buy in buys:
    #     sell = sells[sells > buy]
    #     sell = sell[0] if len(sell) > 0 else sequence_length
    #     buys_optim[buy:sell] = 1
    #
    # buys_optim = buys_optim.float()
    #
    # return buys_optim, buys, sells

    # for seq_i in range(sequence_length):
    #     if li[seq_i]:
    #         tmp = (expected_min_profit_idx[seq_i] == 0) |
    #
    #
    #     else:



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
