# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from model import *
from data import load_all_data
from utils import *
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from optimize import get_wealths
import math

# TODO: reduce repetition
# TODO: get the capitals as well, feed them to the initial states
#       - get also buy prices and timedeltas?
def get_labels(files, coin, n = None, l = 75, c = 1, skew = 0.4, separate = False):
    X = load_all_data(files)

    if n is None:
        n = X.shape[0]

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    diffs = np.diff(np.concatenate([np.array([0]), buys_optim, np.array([0])]))
    idx = np.arange(n + 1)

    buys_li = diffs == 1
    sells_li = diffs == -1

    buys_idx = idx[buys_li]
    sells_idx = idx[sells_li]

    buys_li = buys_li.astype(float)
    sells_li = sells_li.astype(float)

    for i in range(buys_idx.shape[0]):
        buy_i = buys_idx[i]

        start_i = max(0, buy_i - l)
        end_i = min(n, buy_i + l)

        if i > 0:
            start_i = max(start_i, (buy_i + buys_idx[i - 1]) // 2)
            start_i = max(start_i, sells_idx[i - 1])

        if i < buys_idx.shape[0] - 1:
            end_i = min(end_i, (buy_i + sells_idx[i]) // 2)

        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        min_i = min(buy_i - start_i, end_i - start_i - 1)
        min_price = nearby_prices[min_i]
        max_price = np.max(nearby_prices)

        values = (nearby_idx - nearby_idx[min_i]) * (1 - skew * (nearby_idx > nearby_idx[min_i]).astype(float))
        values = np.exp(- c * (max_price / min_price - 1) * values ** 2)
        buys_li[start_i:end_i] = values

    # print(sells_idx[-3:])
    # print(buys_idx[-3:])

    for i in range(sells_idx.shape[0]):
        sell_i = sells_idx[i]

        start_i = max(0, sell_i - l)
        end_i = min(n, sell_i + l)

        if i > 0:
            start_i = max(start_i, (sell_i + sells_idx[i - 1]) // 2)
            start_i = max(start_i, buys_idx[i])

        if i < sells_idx.shape[0] - 1:
            end_i = min(end_i, (sell_i + buys_idx[i + 1]) // 2)

        # print(start_i, end_i)
        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        max_i = min(sell_i - start_i, end_i - start_i - 1)
        # print(start_i, end_i, sell_i, sells_idx.shape[0], i)
        max_price = nearby_prices[max_i]
        min_price = np.min(nearby_prices)

        values = (nearby_idx - nearby_idx[max_i]) * (1 - skew * (nearby_idx > nearby_idx[max_i]).astype(float))
        values = np.exp(- c * (max_price / min_price - 1) * values ** 2)
        sells_li[start_i:end_i] = values

    # buys_optim = (buys_li == 1)

    diffs_li = buys_li - sells_li

    buy = np.zeros(diffs_li.shape[0])
    sell = np.zeros(diffs_li.shape[0])

    li = diffs_li > 0
    buy[li] = diffs_li[li]

    li = diffs_li < 0
    sell[li] = -diffs_li[li]

    buy = buy[:-1]
    sell = sell[:-1]

    do_nothing = 1 - buy - sell

    if separate:
        return buy, sell, do_nothing

    labels = np.stack([buy, sell, do_nothing], axis = 1)

    return labels



def get_labels_test(files, coin, n = None, end_val = 0.001, separate = False):
    X = load_all_data(files)

    if n is None:
        n = X.shape[0]

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    diffs = np.diff(np.concatenate([np.array([0]), buys_optim, np.array([0])]))
    idx = np.arange(n + 1)

    buys_li = diffs == 1
    sells_li = diffs == -1

    buys_idx = idx[buys_li]
    sells_idx = idx[sells_li]

    buys_li = buys_li.astype(float)
    sells_li = sells_li.astype(float)

    for i in range(buys_idx.shape[0]):
        buy_i = buys_idx[i]

        start_i = buy_i
        end_i = sells_idx[i]

        nearby_idx = np.arange(start_i, end_i)

        c = - np.log(end_val) / ((end_i - start_i) ** 2)

        values = np.exp(- c * (nearby_idx - start_i) ** 2)
        buys_li[start_i:end_i] = values

    # print(sells_idx[-3:])
    # print(buys_idx[-3:])

    for i in range(sells_idx.shape[0]):
        sell_i = sells_idx[i]

        start_i = sell_i
        end_i = buys_idx[i + 1] if i < buys_idx.shape[0] - 1 else n

        nearby_idx = np.arange(start_i, end_i)

        c = - np.log(end_val) / ((end_i - start_i) ** 2)

        values = np.exp(- c * (nearby_idx - start_i) ** 2)
        sells_li[start_i:end_i] = values

    buy = buys_li[:-1]
    sell = sells_li[:-1]

    do_nothing = 1 - buy - sell

    if separate:
        return buy, sell, do_nothing

    labels = np.stack([buy, sell, do_nothing], axis = 1)

    return labels




def plot_labels(files, coin, n = 600):
    X = load_all_data(files)

    if n is None:
        n = X.shape[0]

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    wealths, _, _, _, _ = get_wealths(
        X[:n, :], buys_optim
    )

    idx = np.arange(n)

    buy, sell, do_nothing = get_labels(files, coin, n, separate = True)

    plt.style.use('seaborn')
    #plt.plot(buys_optim, label='buys')
    # plt.plot(X[:n, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    # plt.plot(wealths + 1, c='b', alpha=0.5, label='wealth')
    plt.plot((X[:n, 0] / X[0, 0] - 1) * 100, c='k', alpha=0.5, label='price')
    plt.plot(wealths * 100, c='b', alpha=0.5, label='wealth')

    plt.plot(idx, buy, c='g', alpha=0.5)
    plt.plot(idx, sell, c='r', alpha=0.5)

    plt.axhline(0.5, c='k', linestyle=':', alpha=0.75)

    #plt.plot(idx, buys_li - sells_li, c='m', alpha=0.5)

    # plt.yscale('log')
    plt.legend()
    plt.show()



BUY, SELL, DO_NOTHING = 0, 1, 2

def update_state(action, state, price, ma_ref, commissions):
    capital_usd = state[:, 0]
    capital_coin = state[:, 1]
    timedelta = state[:, 2]
    buy_price = state[:, 3]

    li1 = (action == BUY) & (buy_price == -1)
    li2 = (action == SELL) & (buy_price != -1)

    li3 = timedelta != -1
    timedelta[li3] += 1

    timedelta[li1] = 0
    buy_price[li1] = price[li1] / ma_ref[li1]

    timedelta[li2] = -1
    buy_price[li2] = -1

    capital_coin[li1] = capital_usd[li1] / price[li1] * (1 - commissions)
    capital_usd[li1] = 0

    capital_usd[li2] = capital_coin[li2] * price[li2] * (1 - commissions)
    capital_coin[li2] = 0

    state = torch.stack([capital_usd, capital_coin, timedelta, buy_price], dim = 1)

    return state


# TODO: train an ensemble?
def train(coin, files, inputs, params, model, n_epochs, lr, batch_size, sequence_length, print_step, commissions):
    X = load_all_data(files)

    obs, N, ma_ref = get_obs_input(X, inputs, params)
    X = X[-N:, :]

    labels = get_labels(files, coin)

    labels = torch.from_numpy(labels[-N:, :]).type(torch.float32)

    prices = torch.from_numpy(X[:, 0]).type(torch.float32)

    # discard some of the last values; their labels are bad
    N_discard = 20
    obs = obs[:-N_discard, :]
    labels = labels[:-N_discard, :]
    X = X[:-N_discard, :]
    ma_ref = ma_ref[:-N_discard]
    prices = prices[:-N_discard]
    N -= N_discard

    N_test = sequence_length
    # print('N test: {}'.format(N_test))
    N -= N_test

    obs, obs_test = obs[:N, :], obs[N:, :]
    labels, labels_test = labels[:N, :], labels[N:, :]
    X, X_test = X[:N, :], X[N:, :]
    ma_ref, ma_ref_test = ma_ref[:N], ma_ref[N:]
    prices, prices_test = prices[:N], prices[N:]

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(n_epochs):
        i = torch.randint(N - sequence_length, (batch_size,))
        # i = torch.zeros(batch_size).long()# + 1

        # state = init_state(inputs, batch_size = batch_size)
        model.init_state()

        losses = []

        optimizer.zero_grad()

        # for j in range(sequence_length):

        inp = []
        for ii in i:
            inp.append(obs[ii:ii+sequence_length, :])
        inp = torch.stack(inp, dim=1).detach()
        # inp = torch.cat([state, obs[i + j, :]], dim = -1).unsqueeze(0).detach()

        out = model(inp)
        # target = labels[i + j, :]
        target = []
        for ii in i:
            target.append(labels[ii:ii+sequence_length, :])
        target = torch.stack(target, dim=1)

        loss = criterion(out, target)

        loss.backward()
        losses.append(loss.item())

        # state = update_state(out.detach(), state.detach(), prices[i + j], ma_ref[i + j], commissions)

        optimizer.step()

        if e % print_step == 0:
            benchmark_profits = []
            profits = []
            # optim_profits = []

            for b in range(batch_size):
                benchmark_profits.append(X[i[b] + sequence_length - 1, 0] / X[i[b], 0])

                buys = out[:, b, 0].detach().numpy()
                sells = out[:, b, 1].detach().numpy()
                wealths, _, _, _, _ = get_wealths(
                    X[i[b]:i[b]+sequence_length, :], buys, sells, commissions = commissions
                )
                profits.append(wealths[-1] + 1)

            n_round = 4

            print('[Epoch: {}/{}] [Loss: {}] [Avg. Profit: {}] [Benchmark Profit: {}]'.format(
                e,
                n_epochs,
                round_to_n(torch.tensor(losses).mean().item(), n_round),
                round_to_n(torch.tensor(profits).prod().item() ** (1 / batch_size), n_round),
                round_to_n(torch.tensor(benchmark_profits).prod().item() ** (1 / batch_size), n_round),
            ))

    model.eval()
    model.init_state(1)
    inp = obs_test.unsqueeze(1)
    out = model(inp)

    buys = out[:, 0, 0].detach().numpy()
    sells = out[:, 0, 1].detach().numpy()

    initial_usd = 1000

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X_test, buys, sells, initial_usd = initial_usd, commissions = commissions
    )

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[0, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].legend()
    plt.show()

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X_test, buys, sells, initial_usd = initial_usd, commissions = commissions
    )

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[0, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].legend()
    plt.show()




def Qlearn(policy_net, target_net, coin, files, inputs, params, n_epochs, lr, batch_size, sequence_length, print_step, commissions):
    # TODO: move outside of the function?
    # TODO: to lower cpas
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 200 * sequence_length
    TARGET_UPDATE = 10
    initial_usd = 1000
    n_actions = 3

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(10000) # TODO: capacity as parameter

    X = load_all_data(files)

    obs, N, ma_ref = get_obs_input(X, inputs, params)
    X = X[-N:, :]

    prices = torch.from_numpy(X[:, 0]).type(torch.float32)

    # Get a test set
    N_test = sequence_length * 2
    N -= N_test

    obs, obs_test = obs[:N, :], obs[N:, :]
    X, X_test = X[:N, :], X[N:, :]
    ma_ref, ma_ref_test = ma_ref[:N], ma_ref[N:]
    prices, prices_test = prices[:N], prices[N:]

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    def select_action(state, steps_done):
        sample = torch.rand(batch_size)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        li = sample > eps_threshold

        action = torch.randint(n_actions, (batch_size,))
        if li.any():
            action[li] = policy_net(state[li, :]).max(1)[1].view(-1)

        return action, steps_done

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.uint8)
        if non_final_mask.any():
            non_final_next_states = torch.stack([s for s in batch.next_state
                                                        if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch)[:, action_batch].diag()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        if non_final_mask.any():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    for e in range(n_epochs):
        sub_state = init_state(inputs, batch_size, initial_usd)

        t = torch.randint(1, N - sequence_length, (batch_size,))
        # t = torch.zeros(batch_size).long()# + 1

        prev_state = torch.cat([sub_state, obs[t - 1, :]], dim = 1)
        state = torch.cat([sub_state, obs[t, :]], dim = 1)
        profits = []

        for seq_i in range(sequence_length):
            # Select and perform an action
            action, steps_done = select_action(state, steps_done)

            # Calculate reward
            reward = (sub_state[:, 0] + sub_state[:, 1] * prices[t + seq_i]) / initial_usd
            profits.append(reward.prod() ** (1 / batch_size))
            reward = torch.log(reward)

            # Observe new state
            if seq_i < sequence_length - 1:
                sub_state = update_state(action, sub_state, prices[t + seq_i], ma_ref[t + seq_i], commissions)
                next_state = torch.cat([sub_state, obs[t + seq_i, :]], dim = 1)
            else:
                next_state = None

            # Store the transitions in memory
            for b in range(batch_size):
                memory.push(
                    state[b],
                    action[b],
                    next_state[b] if next_state is not None else None,
                    reward[b]
                )

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if e % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        profits = torch.stack(profits)

        if e % print_step == 0:
            n_round = 4

            print('[Epoch: {}/{}] [Loss: {}] [Avg. Profit: {}] [Eps: {}]'.format(
                e,
                n_epochs,
                round_to_n(loss, n_round),
                round_to_n(profits[-1].item(), n_round),
                round_to_n(EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY))
            ))

    policy_net.eval()
    sub_state = init_state(inputs, 1, initial_usd)

    state = torch.cat([sub_state, obs_test[0, :].view(1, -1)], dim = 1)
    profits = []
    actions = []

    for seq_i in range(N_test):
        # Select and perform an action
        action = policy_net(state).max(1)[1].view(-1)
        actions.append(action)

        # Calculate reward
        reward = (sub_state[:, 0] + sub_state[:, 1] * prices_test[seq_i]) / initial_usd
        profits.append(reward.prod())

        # print(prices_test[seq_i], prices_test[seq_i].shape)
        # Observe new state
        if seq_i < N_test - 1:
            sub_state = update_state(action, sub_state, prices_test[seq_i].view(-1), ma_ref_test[seq_i].view(-1), commissions)
            next_state = torch.cat([sub_state, obs_test[seq_i, :].view(1, -1)], dim = 1)
        else:
            next_state = None

        # Move to the next state
        state = next_state


    profits = torch.stack(profits).numpy()
    actions = torch.stack(actions).numpy()

    plt.style.use('seaborn')

    plt.plot(X_test[:, 0] / X_test[0, 0], label='price')
    plt.plot(profits, label='profit')
    plt.legend()
    plt.show()

    counts = [
        (actions == BUY).astype(float).sum(),
        (actions == SELL).astype(float).sum(),
        (actions == DO_NOTHING).astype(float).sum(),
    ]

    ticks = ['BUY', 'SELL', 'DO_NOTHING']
    x_pos = np.arange(3)

    plt.bar(x_pos, counts, align='center', alpha=0.75)
    plt.xticks(x_pos, ticks)
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    commissions = 0.00075

    # inputs:
    #   note: all prices (and stds) are relative to a running average price
    #   - state:
    #       - capital_usd
    #       - capital_coin (in usd)
    #       - time since bought (or -1)
    #       - price when bought (or -1)
    #       - other data at time of bought?
    #   - obs:
    #       - close, high, low, open (not all?)
    #       - running std, or bollinger band width
    #       - sma, alligator stuff?
    #       - smoothed returns
    #       - stoch
    #       - ha
    inputs = {
        # states
        # 'capital_usd': 1,
        # 'capital_coin': 1,
        # 'timedelta': 1,
        # 'buy_price': 1,
        # obs
        'price': 4,
        'mus': 3,
        'std': 3,
        'ma': 3,
        'ha': 4,
        'stoch': 3,
    }

    params = {
        'alpha': 0.8,
        'std_window_min_max': [30, 2000],
        'ma_window_min_max': [30, 2000],
        'stoch_window_min_max': [30, 2000],
    }

    sequence_length = 500

    lr = 0.0005
    batch_size = 128

    # NN model definition
    policy_net = FFN(inputs, batch_size, use_lstm = True, Qlearn = False)
    target_net = FFN(inputs, batch_size, use_lstm = False, Qlearn = True)

    n_epochs = 350
    print_step = max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    plot_labels(files, coin)

    start_time = time.time()

    train(
        coin = coin,
        files = files,
        inputs = inputs,
        params = params,
        model = policy_net,
        n_epochs = n_epochs,
        lr = lr,
        batch_size = batch_size,
        sequence_length = sequence_length,
        print_step = print_step,
        commissions = commissions,
    )

    policy_net.Qlearn = True

    # Qlearn(
    #     policy_net = policy_net,
    #     target_net = target_net,
    #     coin = coin,
    #     files = files,
    #     inputs = inputs,
    #     params = params,
    #     n_epochs = n_epochs,
    #     lr = lr,
    #     batch_size = batch_size,
    #     sequence_length = sequence_length,
    #     print_step = print_step,
    #     commissions = commissions,
    # )

    print('Time taken: {} seconds'.format(round_to_n(time.time() -start_time, 3)))
