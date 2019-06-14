import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor

def calc_actions(model, X, batch_size, input_length, latency, initial_capital = 1000, commissions = 0.00075):
    means = np.reshape(np.mean(X, axis=0), (1,6))
    stds = np.reshape(np.std(X, axis=0), (1,6))

    capital_usd = initial_capital
    capital_coin = 0

    wealths = [initial_capital]
    capital_usds = [capital_usd]
    capital_coins = [capital_coin]
    buy_amounts = []
    sell_amounts = []

    for i in range(batch_size):
        inp = np.reshape((X[i:i+input_length, :] - means) / stds, (1, input_length, 6))
        BUY, SELL, DO_NOTHING, amount = tuple(np.reshape(model.predict(inp), (4,)))
        price = X[i + input_length + latency - 1, 0]

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
    # price = X[-1, 0]
    # std = np.std(wealths)
    reward = wealths[-1] #/ (std if std > 0 else 1)
    # reward = (wealths[-1] - (price / X[input_length, 0] - 1)) / (std * (price_max - price_min))
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

def calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital):
    metrics = {}
    metrics['reward'] = reward
    metrics['profit'] = wealths[-1]
    metrics['max_profit'] = np.max(wealths)
    metrics['min_profit'] = np.min(wealths)
    metrics['buys'] = np.sum(buy_amounts) / initial_capital
    metrics['sells'] = np.sum(sell_amounts) / initial_capital
    metrics['std'] = np.std(wealths)
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
