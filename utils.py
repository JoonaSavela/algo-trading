import numpy as np
import matplotlib.pyplot as plt

# TODO: do something if there are too many zeros
def is_valid_interval(data, left = 0, right = None):
    right = len(data) if right is None else right
    return not np.any(data[left:right] == 0)

# TODO: do something if there are too many zeros
def fill_zeros(data):
    new_data = data
    n = len(data)
    for i in range(n):
        if data[i] == 0:
            if i == 0:
                k = i + 1
                while k < n and data[k] == 0:
                    k += 1
                if k < n:
                    new_data[i] = data[k]
            elif i == n - 1:
                j = i - 1
                while j >= 0 and data[j] == 0:
                    j -= 1
                if j >= 0:
                    new_data[i] = data[j]
            else:
                k = i + 1
                while k < n and data[k] == 0:
                    k += 1
                j = i - 1
                while j >= 0 and data[j] == 0:
                    j -= 1
                if k < n and j >= 0:
                    w1 = k - i
                    w2 = i - j
                    new_data[i] = (w1 * data[k] + w2 * data[j]) / (w1 + w2)

    return new_data

def normalize_values(opens, highs, lows, closes, volumes):
    i = 0
    while i < opens.size and opens[i] == 0:
        i += 1
    divider = opens[i] if i < opens.size else 1
    return opens / divider, highs / divider, lows / divider, closes / divider, volumes / 100000

def calculate_y(closes):
    close_min = closes.min()
    close_max = closes.max()
    if close_max == close_min:
        print(closes)
    current = closes[0]
    peak = 2 * (current - close_min) / (close_max - close_min) - 1
    percent_gain = close_max / current - 1
    percent_loss = close_min / current - 1
    return np.array([peak, percent_gain, percent_loss])

def transform_data(opens, highs, lows, closes, volumes, input_length, y_length = 3):
    n = len(opens) - 2 * input_length + 2 # number of rows
    if n <= 0 or np.all(closes == closes[0]):
        return np.empty(shape=[0, input_length * 5]), np.empty(shape=[0, y_length])
    X = np.zeros(shape=[n, input_length * 5]) # 5 = number of types of input variables
    Y = np.zeros(shape=[n, y_length])
    li = np.ones(n, dtype=bool) # hotfix for the only one value bug

    i = input_length - 1

    while i <= len(opens) - input_length:
        b = i + 1
        a = b - input_length
        c = i
        d = c + input_length

        x = np.concatenate([opens[a:b], highs[a:b], lows[a:b], closes[a:b], volumes[a:b]])
        X[a,:] = x
        Y[a,:] = np.reshape(calculate_y(closes[c:d]), [1, y_length])
        if np.all(closes[c:d] == closes[c]):
            li[a] = False

        i += 1

    return X[li,:], Y[li,:]

def plot_y(X, Y, X_col):
    tmp = X[:,X_col]
    size = 8

    fig, axes = plt.subplots(figsize=(16, 4), ncols=Y.shape[1])

    for i in range(len(axes)):
        axis = axes[i]
        plot = axis.scatter(range(len(tmp)), tmp, c=Y[:,i], cmap=plt.get_cmap('viridis'), s=size)
        fig.colorbar(plot, ax=axis)

    plt.show()
