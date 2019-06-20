import matplotlib.pyplot as plt
from utils import plot_y
from model import RelationalMemory
from data import get_and_save_all
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy

def main():
    A = torch.randn((52))
    r = torch.randn((52))

    permute_indices = list(range(len(A.shape)))[::-1]

    delta = 1 * \
        torch.tensordot(A.permute(*permute_indices), r, dims=1)

    print(delta.shape)

    try:
        delta = delta.permute(*permute_indices[1:])
    except TypeError:
        pass

    print(delta.shape)

    print(*permute_indices[1:])

    # AT = A.permute(*permute_indices)

    # print(AT.shape)
    # print(torch.tensordot(AT, r, dims=1).permute(*permute_indices[1:]).shape)
    # print(A[0, :, :] * r[0] ==

    # input_size = 6
    # seq_length = 1
    #
    # mem_slots = 4
    # num_heads = 2
    #
    # model = RelationalMemory(mem_slots=mem_slots, head_size=input_size, input_size=input_size, num_heads=num_heads, num_blocks=1, forget_bias=1., input_bias=0.)
    # model_memory = model.initial_state(batch_size=1)
    #
    #
    # noise = []
    # for i in range(5):
    #     x = {}
    #     for k, w in model.state_dict().items():
    #         x[k] = torch.randn(w.shape)
    #     noise.append(x)
    #
    # state_dict_try = copy.deepcopy(model.state_dict())
    # for k, i in noise[0].items():
    #     jittered = 1*i
    #     state_dict_try[k] += jittered
    #
    # print(state_dict_try)
    # print()
    # print(model.state_dict())
    #
    # model.load_state_dict(state_dict_try)
    #
    # print()
    # print(state_dict_try)
    # print()
    # print(model.state_dict())



    # for filename in glob.glob('run_summaries/*.csv'):
    #     print(filename)
    #     table = pd.read_csv(filename)
    #     n = table.shape[0]
    #     print(n)
    #
    #     print(np.prod(table['profit'] + 1))
    #     print(np.power(np.prod(table['profit'] + 1), 1/n))
    #     print(np.prod(table['max_profit'] + 1))
    #     print(np.power(np.prod(table['max_profit'] + 1), 1/n))
    #     print(np.prod(table['min_profit'] + 1))
    #     print(np.power(np.prod(table['min_profit'] + 1), 1/n))
    #
    #     plt.plot(table['iteration'], table['profit'])
    #     plt.plot(table['iteration'], table['max_profit'])
    #     plt.plot(table['iteration'], table['min_profit'])
    #     plt.show()
    #
    #     plt.plot(table['iteration'], table['reward'])
    #     plt.show()

    # for filename in glob.glob('data/*/*.json'):
    #     with open(filename, 'r') as file:
    #         obj = json.load(file)
    #     data = obj['Data']
    #
    #     X = np.zeros(shape=(len(data), 6))
    #     for i in range(len(data)):
    #         item = data[i]
    #         tmp = []
    #         for key, value in item.items():
    #             if key != 'time':
    #                 tmp.append(value)
    #         X[i, :] = tmp
    #
    #     plt.plot(range(len(data)), X[:, 0] / X[0, 0])
    #     plt.title(filename)
    #     plt.show()


if __name__ == "__main__":
    main()
