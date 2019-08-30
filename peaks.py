import torch
import torch.nn as nn
from scipy.signal import find_peaks, peak_prominences
import glob
from utils import *
from data import *
from model import *
import numpy as np
from optimize import get_wealths
import matplotlib.pyplot as plt

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def get_left_prominence(x, wlen = None):
    if wlen is None:
        wlen = x.shape[0]
    wlen = min(wlen, x.shape[0])

    x = x[-wlen:]

    end_i = wlen - 1

    li = x >= x[end_i]

    start_i = end_i

    while li[start_i] and start_i > 0:
        start_i -= 1

    while not li[start_i] and start_i > 0:
        start_i -= 1

    prominence = x[end_i] - np.min(x[start_i:end_i + 1])

    return prominence

def get_left_prominences(x, wlen = 60):
    N = x.shape[0] - wlen + 1
    prominences = np.zeros(N)

    for i in range(N):
        prominences[i] = get_left_prominence(x[i:i+wlen])

    return prominences


def get_approx_peaks(sells):
    lag = 40
    threshold = 0.1
    influence = 0.125
    result = thresholding_algo(sells, lag, threshold, influence)
    diffs = np.diff(sells, 2)

    wlen = 60
    sell_prominences = get_left_prominences(sells, wlen = wlen)
    buy_prominences = get_left_prominences(1 - sells, wlen = wlen)

    # buy_li = (result['signals'][2:] == -1) & (diffs[0:] > 0) #& (diffs[:-1] < 0)
    # sell_li = (result['signals'][2:] == 1) & (diffs[0:] < 0) #& (diffs[:-1] > 0)

    th = 0.05
    buy_li = (buy_prominences > th) & (result['signals'][wlen - 1:] == -1) & (diffs[wlen - 1 - 2:] > 0)
    sell_li = (sell_prominences > th) & (result['signals'][wlen - 1:] == 1) & (diffs[wlen - 1 - 2:] < 0)

    w = 60*2
    sells = stochastic_oscillator(np.repeat(sells.reshape(-1, 1), 4, axis = 1), w)
    N = sells.shape[0]

    th = 0.05
    sell_li = (sell_li[-N:]) & (sells >= 1 - th)
    buy_li = (buy_li[-N:]) & (sells <= th)

    waiting_time = 0

    idx = np.arange(0, sells.shape[0]) + waiting_time
    # idx = np.arange(wlen - 1, sells.shape[0])

    return idx[~sell_li & buy_li], idx[~buy_li & sell_li], N


def plot_peaks(files, inputs, params, model, sequence_length):
    X = load_all_data(files)
    returns = X[1:, 0] / X[:-1, 0] - 1

    alpha = 0.#85
    mus = smoothed_returns(X, alpha=alpha)
    mus = smoothed_returns(np.cumprod(mus + 1).reshape(-1, 1), alpha=alpha)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]
    returns = returns[-N:]
    mus = mus[-N:]

    model.init_state()

    inp = obs.unsqueeze(1)
    out = model(inp).squeeze(1)

    buys = out[:, 0].detach().numpy()
    sells = out[:, 1].detach().numpy()

    buy_peaks_approx, sell_peaks_approx, N = get_approx_peaks(sells)

    X = X[-N:, :]
    returns = returns[-N:]
    mus = mus[-N:]
    sells = sells[-N:]
    buys = buys[-N:]

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    buy_peaks, sell_peaks = get_peaks(sells)

    sequence_length = min(sequence_length, N)

    X = X[:sequence_length, :]
    returns = returns[:sequence_length]
    mus = mus[:sequence_length]
    sells = sells[:sequence_length]

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    d = 0
    buy_peaks += d
    sell_peaks += d
    buy_peaks = buy_peaks[buy_peaks < sequence_length]
    sell_peaks = sell_peaks[sell_peaks < sequence_length]
    buy_peaks_approx = buy_peaks_approx[buy_peaks_approx < sequence_length]
    sell_peaks_approx = sell_peaks_approx[sell_peaks_approx < sequence_length]

    owns1 = np.zeros((sequence_length,))

    for peak in buy_peaks:
        sell_peak = sell_peaks[sell_peaks > peak]
        sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
        owns1[peak:sell_peak] = 1

    buys1 = owns1 == 1
    sells1 = owns1 == 0

    buys2 = returns > 0
    sells2 = returns < 0

    buys1 = buys1 & buys2
    sells1 = sells1 & sells2

    buys1 = buys1.astype(float)
    sells1 = sells1.astype(float)

    wealths1, _, _, _, _ = get_wealths(
        X, buys1, sells1
    )

    print(wealths1[-1] + 1)

    owns2 = np.zeros((sequence_length,))

    for peak in buy_peaks_approx:
        sell_peak = sell_peaks_approx[sell_peaks_approx > peak]
        sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
        owns2[peak:sell_peak] = 1

    buys2 = owns2 == 1
    sells2 = owns2 == 0

    buys3 = mus > 0
    sells3 = mus < 0

    buys2 = buys2 & buys3
    sells2 = sells2 & sells3

    buys2 = buys2.astype(float)
    sells2 = sells2.astype(float)

    wealths2, _, _, _, _ = get_wealths(
        X, buys2, sells2
    )

    print(wealths2[-1] + 1)

    #lag = 15
    #threshold = 4
    #influence = 0.5
    #result = thresholding_algo(sells, lag, threshold, influence)
    #buys_approx = result['signals']
    #print(buys_approx.shape, buy_peaks.shape)


    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'mo', alpha=0.7, label='sell peaks')
    ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'co', alpha=0.7, label='buy peaks')
    ax[0].plot(sell_peaks_approx, X[sell_peaks_approx, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks approx')
    ax[0].plot(buy_peaks_approx, X[buy_peaks_approx, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks approx')
    # ax[0].plot(np.cumprod(mus + 1), alpha=0.7, label='smoothed price')
    ax[0].plot(wealths1 + 1, alpha=0.7, label='wealth1')
    ax[0].plot(wealths2 + 1, alpha=0.7, label='wealth2')
    ax[0].legend()

    #ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].plot(sell_peaks, sells[sell_peaks], 'mo', alpha=0.7, label='sell peaks')
    ax[1].plot(buy_peaks, sells[buy_peaks], 'co', alpha=0.7, label='buy peaks')
    #ax[1].plot(np.arange(wlen - 1, sequence_length), sell_prominences, 'k', alpha=0.6, label='sell prominences')
    #ax[1].plot(np.arange(wlen - 1, sequence_length), buy_prominences, 'g', alpha=0.6, label='buy prominences')
    ax[1].plot(buy_peaks_approx, sells[buy_peaks_approx], 'go', alpha=0.5, label='buy peaks approx')
    ax[1].plot(sell_peaks_approx, sells[sell_peaks_approx], 'ro', alpha=0.5, label='sell peaks approx')
    ax[1].legend()
    plt.show()


def get_labels(sells, use_tanh):
    # TODO: calculate these; save into file or something
    returns_dict = {
        -9: 1.0919042700491126,
        -8: 1.2247738109953097,
        -7: 1.490936256673261,
        -6: 1.70637431678199,
        -5: 2.0100570566933804,
        -4: 2.110587760602813,
        -3: 2.1718015504678863,
        -2: 2.1546475450636637,
        -1: 2.0275320009700644,
        0: 1.862710155114335,
        1: 1.4483548475933,
        2: 1.1213635900726804,
        3: 1.0507336634485362,
    }

    buy_peaks, sell_peaks = get_peaks(sells)

    N = sells.shape[0]

    buy_labels = np.zeros(N)
    sell_labels = np.zeros(N)

    max_profit = max(list(returns_dict.values()))
    #print(max_profit)

    d = 3

    for peak in buy_peaks:
        for k, v in returns_dict.items():
            if peak + k + d < N:
                buy_labels[peak + k + d] = (v - 1) / (max_profit - 1)

    for peak in sell_peaks:
        for k, v in returns_dict.items():
            if peak + k + d < N:
                sell_labels[peak + k + d] = (v - 1) / (max_profit - 1)

    diffs_labels = buy_labels - sell_labels
    if use_tanh:
        return diffs_labels.reshape((-1, 1))

    buy = np.zeros(N)
    sell = np.zeros(N)

    li = diffs_labels > 0
    buy[li] = diffs_labels[li]

    #print(buy.mean())

    li = diffs_labels < 0
    sell[li] = -diffs_labels[li]

    do_nothing = 1 - buy - sell

    labels = np.stack([buy, sell, do_nothing], axis = 1)

    return labels

def train(files, inputs, params, model, signal_model, sequence_length, n_epochs, batch_size, lr, commissions, print_step, use_tanh, eps, save):
    X = load_all_data(files)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]

    signal_model.init_state()

    inp = obs.unsqueeze(1)
    signal_out = signal_model(inp).squeeze(1)

    sells = signal_out[:, 1].detach().numpy()

    labels = get_labels(sells, use_tanh)
    labels = torch.from_numpy(labels).type(torch.float32)

    N_test = sequence_length
    N -= N_test

    signal_out, signal_out_test = signal_out[:N, :], signal_out[N:, :]
    labels, labels_test = labels[:N, :], labels[N:, :]
    X, X_test = X[:N, :], X[N:, :]

    if use_tanh:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(n_epochs):
        t = torch.randint(N - sequence_length, (batch_size,))
        #t = torch.zeros(batch_size).long()

        model.init_state()

        losses = []

        optimizer.zero_grad()

        inp = []
        for tt in t:
            inp.append(signal_out[tt:tt+sequence_length, :])
        inp = torch.stack(inp, dim=1).detach()

        out = model(inp)

        target = []
        for tt in t:
            target.append(labels[tt:tt+sequence_length, :])
        target = torch.stack(target, dim=1)

        loss = criterion(out, target)

        loss += std_loss(out, sequence_length, batch_size, eps)

        loss += diff_loss(out, batch_size, use_tanh, e, n_epochs)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if e % print_step == 0:
            profits = []

            out = out.detach().numpy()

            for b in range(batch_size):
                if use_tanh:
                    buys = np.zeros(sequence_length)
                    sells = np.zeros(sequence_length)

                    li = out[:, b, 0] > 0
                    buys[li] = out[li, b, 0]

                    li = out[:, b, 0] < 0
                    sells[li] = -out[li, b, 0]
                else:
                    buys = out[:, b, 0]
                    sells = out[:, b, 1]
                wealths, _, _, _, _ = get_wealths(
                    X[t[b]:t[b]+sequence_length, :], buys, sells, commissions = commissions
                )
                profits.append(wealths[-1] + 1)

            n_round = 4

            print('[Epoch: {}/{}] [Loss: {}] [Avg. Profit: {}]'.format(
                e,
                n_epochs,
                round_to_n(torch.tensor(losses).mean().item(), n_round),
                round_to_n(torch.tensor(profits).prod().item() ** (1 / batch_size), n_round),
            ))

    model.eval()
    model.init_state(1)
    inp = signal_out_test.unsqueeze(1)
    out = model(inp).detach().numpy()

    if use_tanh:
        buys = np.zeros(sequence_length)
        sells = np.zeros(sequence_length)

        li = out[:, 0, 0] > 0
        buys[li] = out[li, 0, 0]

        li = out[:, 0, 0] < 0
        sells[li] = -out[li, 0, 0]
    else:
        buys = out[:, 0, 0]
        sells = out[:, 0, 1]

    initial_usd = 1000

    print(buys.max(), buys.min())
    print(sells.max(), sells.min())

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X_test, buys, sells, initial_usd = initial_usd, commissions = commissions
    )

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[-1, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(signal_out_test[:, 1].detach().numpy(), c='k', alpha=0.5, label='signal')
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

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[-1, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(signal_out_test[:, 1].detach().numpy(), c='k', alpha=0.5, label='signal')
    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].legend()
    plt.show()

    buys = buys ** 2
    sells = sells ** 2

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X_test, buys, sells, initial_usd = initial_usd, commissions = commissions
    )

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[-1, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(signal_out_test[:, 1].detach().numpy(), c='k', alpha=0.5, label='signal')
    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].legend()
    plt.show()

    if save:
        torch.save(model.state_dict(), 'models/' + model.name + '.pt')



if __name__ == '__main__':

    # inputs = {
    #     'price': 4,
    #     'mus': 3,
    #     'std': 3,
    #     'ma': 3,
    #     'ha': 4,
    #     'stoch': 3,
    # }

    inputs = {
        'price': 4,
        'mus': 4,
        'std': 4,
        'ma': 4,
        'ha': 4,
        'stoch': 4,
    }
    hidden_size = sum(list(inputs.values())) * 2


    params = {
        'alpha': 0.8,
        'std_window_min_max': [30, 2000],
        'ma_window_min_max': [30, 2000],
        'stoch_window_min_max': [30, 2000],
    }

    sequence_length = 500000

    signal_model = FFN(inputs, 1, use_lstm = True, Qlearn = False, use_tanh = False, hidden_size = hidden_size)
    signal_model.load_state_dict(torch.load('models/' + signal_model.name + '.pt'))
    signal_model.eval()

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    plot_peaks(files[:200], inputs, params, signal_model, sequence_length)

    batch_size = 128
    hidden_size = 16
    n_epochs = 1000
    print_step = max(n_epochs // 20, 1)
    lr = 0.0005
    sequence_length = 500
    commissions = 0.00075
    use_tanh = True
    eps = 1e-2
    save = True

    model = FFN(dict(n_inputs=3), batch_size, use_tanh = use_tanh, hidden_size = hidden_size)

    train(
        files = files,
        inputs = inputs,
        params = params,
        model = model,
        signal_model = signal_model,
        sequence_length = sequence_length,
        n_epochs = n_epochs,
        batch_size = batch_size,
        lr = lr,
        commissions = commissions,
        print_step = print_step,
        use_tanh = use_tanh,
        eps = eps,
        save = save,
    )
