import numpy as np
from math import log10, floor
try:
    import torch
except ImportError:
    print('Could not import torch')

def std(X, window_size):
    res = []
    for i in range(X.shape[0] - window_size + 1):
        res.append(np.std(X[i:i+window_size, 0]))
    return np.array(res)

def sma(X, window_size):
    return np.convolve(X[:, 0], np.ones((window_size,))/window_size, mode='valid')

def stochastic_oscillator(X, window_size = 3 * 14, k = 1, latency = 0):
    res = []
    for i in range(X.shape[0] - window_size + 1 - latency):
        max_price = np.max(X[i:i + window_size, 1])
        min_price = np.min(X[i:i + window_size, 2])
        min_close = np.min(X[i:i + window_size, 0])
        stoch = (X[i + window_size - 1, 0] - min_close) / (max_price - min_price)
        res.append(stoch)
    res = np.array(res)
    res = np.convolve(res, np.ones((k,))/k, mode='valid')
    return res

def heikin_ashi(X):
    res = np.zeros((X.shape[0] - 1, 4))
    for i in range(X.shape[0] - 1):
        ha_close = 0.25 * np.sum(X[i + 1, :4])
        if i == 0:
            ha_open = 0.5 * (X[0, 0] + X[0, 3])
        else:
            ha_open = 0.5 * np.sum(res[i - 1, :2])
        ha_high = np.max([X[i + 1, 1], ha_close, ha_open])
        ha_low = np.min([X[i + 1, 2], ha_close, ha_open])
        res[i, :] = [ha_close, ha_open, ha_high, ha_low]
    return res

def calc_actions(model, X, sequence_length, latency, window_size, k, initial_capital = 1000, commissions = 0.00075):
    capital_usd = initial_capital
    capital_coin = 0

    wealths = [initial_capital]
    # capital_usds = [capital_usd]
    # capital_coins = [capital_coin]
    buy_amounts = []
    sell_amounts = []

    # model_memory = model.initial_state(batch_size=1)

    stoch = stochastic_oscillator(X, window_size, k)

    tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
    ma = sma(tp, window_size)
    stds = std(tp, window_size)

    X = X[-sequence_length:, :]
    ma = ma[-sequence_length:] / X[0, 0] - 1
    stds = stds[-sequence_length:] / X[0, 0]

    memory = model.initial_state(batch_size = 1)

    for i in range(sequence_length):
        inp = X[i, :4] / X[0, :4] - 1
        # inp = X[i, :] / X[0, :] - 1
        inp = np.append(inp, [stoch[i] * 2 - 1, ma[i], stds[i]])
        inp = torch.from_numpy(inp.astype(np.float32)).view(1, 1, -1)
        logits, memory = model(inp, memory)
        BUY, SELL, DO_NOTHING = tuple(logits.view(3).data.numpy())
        price = X[i + latency, 0]

        amount_coin_buy = BUY * capital_usd / price * (1 - commissions)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * price * (1 - commissions)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        buy_amounts.append(amount_usd_buy)
        sell_amounts.append(amount_usd_sell)

        wealths.append(capital_usd + capital_coin * price)
        # capital_usds.append(capital_usd)
        # capital_coins.append(capital_coin * price)

    price = X[-1, 0]
    amount_usd = capital_coin * price * (1 - commissions)
    capital_usd += amount_usd
    sell_amounts.append(amount_usd)
    capital_coin = 0

    wealths.append(capital_usd)
    # capital_usds.append(capital_usd)
    # capital_coins.append(capital_coin * price)

    wealths = np.array(wealths) / wealths[0] - 1
    # capital_usds = np.array(capital_usds) / initial_capital
    # capital_coins = np.array(capital_coins) / initial_capital

    return wealths, buy_amounts, sell_amounts

def calc_reward(wealths, buy_amounts, initial_capital):
    reward = np.average(wealths, weights=range(wealths.shape[0])) / (np.sum(buy_amounts) / initial_capital + 1)
    return reward

def round_to_n(x, n = 2):
    if x == 0: return x
    res = round(x, -int(floor(log10(abs(x)))) + (n - 1)) if x != 0 else 0
    res = int(res) if abs(res) >= 10**(n - 1) else res
    return res

def floor_to_n(x, n = 2):
    if x == 0: return x
    p = -int(floor(log10(abs(x)))) + (n - 1)
    res = floor(x * 10 ** p) / 10 ** p
    res = int(res) if abs(res) >= 10**(n - 1) else res
    return res

def calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital, first_price, last_price):
    metrics = {}
    metrics['reward'] = reward
    metrics['profit'] = wealths[-1]
    metrics['max_profit'] = np.max(wealths)
    metrics['min_profit'] = np.min(wealths)
    metrics['buys'] = np.sum(buy_amounts) / initial_capital
    metrics['sells'] = np.sum(sell_amounts) / initial_capital
    metrics['benchmark_profit'] = last_price / first_price - 1
    for key, value in metrics.items():
        metrics[key] = round_to_n(value)
    return metrics

def get_time(filename):
    split1 = filename.split('/')
    split2 = split1[2].split('.')
    return int(split2[0])
