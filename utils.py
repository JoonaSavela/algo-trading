import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
import torch

def calc_actions(model, X, sequence_length, latency, initial_capital = 1000, commissions = 0.00075):
    capital_usd = initial_capital
    capital_coin = 0

    wealths = [initial_capital]
    capital_usds = [capital_usd]
    capital_coins = [capital_coin]
    buy_amounts = []
    sell_amounts = []

    model_memory = model.initial_state(batch_size=1)

    for i in range(sequence_length):
        inp = torch.from_numpy((X[i, :] / X[0, :] - 1).astype(np.float32)).view(1, 1, -1)
        logit, _, model_memory = model(inp, model_memory)
        BUY, SELL, DO_NOTHING, amount = tuple(logit.view(4).data.numpy())
        price = X[i + latency, 0]

        if BUY > SELL and BUY > DO_NOTHING:
            amount_coin = amount * capital_usd / price * (1 - commissions)
            capital_coin += amount_coin
            capital_usd *= 1 - amount
            buy_amounts.append(amount_coin * price)
        elif SELL > BUY and SELL > DO_NOTHING:
            amount_usd = amount * capital_coin * price * (1 - commissions)
            capital_usd += amount_usd
            capital_coin *= 1 - amount
            sell_amounts.append(amount_usd)

        wealths.append(capital_usd + capital_coin * price)
        capital_usds.append(capital_usd)
        capital_coins.append(capital_coin)

    price = X[-1, 0]
    amount_usd = capital_coin * price * (1 - commissions)
    capital_usd += amount_usd
    sell_amounts.append(amount_usd)
    capital_coin = 0

    wealths.append(capital_usd)
    capital_usds.append(capital_usd)
    capital_coins.append(capital_coin)

    wealths = np.array(wealths) / wealths[0] - 1

    return wealths, buy_amounts, sell_amounts

def calc_reward(wealths):
    reward = np.mean(wealths)
    return reward

def round_to_n(x, n = 2):
    res = round(x, -int(floor(log10(abs(x)))) + (n - 1)) if x != 0 else 0
    res = int(res) if abs(res) >= 10**(n - 1) else res
    return res

def floor_to_n(x, n = 2):
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

def plot_y(X, Y, X_col):
    tmp = X[:,X_col]
    size = 8

    fig, axes = plt.subplots(figsize=(16, 4), ncols=Y.shape[1])

    for i in range(len(axes)):
        axis = axes[i]
        plot = axis.scatter(range(len(tmp)), tmp, c=Y[:,i], cmap=plt.get_cmap('viridis'), s=size)
        fig.colorbar(plot, ax=axis)

    plt.show()
