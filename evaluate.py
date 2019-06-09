from model import build_model
import numpy as np
import glob
import matplotlib.pyplot as plt
import json

batch_size = 60*4
input_length = 4 * 14
latency = 1
commissions = 0#0.00075

def evaluate():
    model = build_model()
    filename = 'models/model_weights.h5'
    model.load_weights(filename)
    filename = np.random.choice(glob.glob('data/*/*.json'))
    print(filename)
    with open(filename, 'r') as file:
        obj = json.load(file)

    start_index = np.random.choice(len(obj['Data']) - batch_size - input_length - latency)

    data = obj['Data'][start_index:start_index + batch_size + input_length + latency]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key, value in item.items():
            if key != 'time':
                tmp.append(value)
        X[i, :] = tmp

    means = np.reshape(np.mean(X, axis=0), (1,6))
    stds = np.reshape(np.std(X, axis=0), (1,6))

    initial_capital = 1000
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
            # print('BUY:', amount, price)
            capital_coin += amount * capital_usd / price * (1 - commissions)
            capital_usd *= 1 - amount
            buy_amounts.append(amount)
        elif SELL > BUY and SELL > DO_NOTHING:
            # print('SELL:', amount, price)
            capital_usd += amount * capital_coin * price * (1 - commissions)
            capital_coin *= 1 - amount
            sell_amounts.append(amount)

        wealths.append(capital_usd + capital_coin * price)
        capital_usds.append(capital_usd)
        capital_coins.append(capital_coin)

    price = X[-1, 0]
    # print('SELL:', 1.0, price)
    capital_usd += capital_coin * price
    capital_coin = 0

    wealths.append(capital_usd + capital_coin * price)
    capital_usds.append(capital_usd)
    capital_coins.append(capital_coin)
    sell_amounts.append(1)

    wealths = np.array(wealths) / wealths[0] - 1
    # plt.plot(range(batch_size + 2), wealths)
    # plt.show()
    std = np.std(wealths)
    reward = wealths[-1] / (std if std > 0 else 1)
    # print(reward)

    metrics = {}
    metrics['reward'] = reward
    metrics['profit'] = wealths[-1]
    metrics['max_profit'] = np.max(wealths)
    metrics['min_profit'] = np.min(wealths)
    metrics['std'] = np.std(wealths)

    print(metrics)
    print(len(buy_amounts))
    print(len(sell_amounts))

    plt.plot(range(batch_size), X[input_length:input_length+batch_size,0] / X[input_length, 0])
    plt.show()

    plt.plot(range(len(wealths)), wealths)
    plt.show()

    plt.plot(range(len(capital_usds)), capital_usds)
    plt.plot(range(len(capital_coins)), np.array(capital_coins) * X[input_length,0])
    plt.show()


if __name__ == '__main__':
    evaluate()
