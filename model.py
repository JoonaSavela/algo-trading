import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def build_model(input_length, alpha = 0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_length * 2, activation='relu', input_dim=input_length),
        tf.keras.layers.Dense(input_length * 4, activation='relu'),
        tf.keras.layers.Dense(input_length * 2, activation='relu'),
        tf.keras.layers.Dense(3, activation='tanh')
    ])

    optimizer = tf.keras.optimizers.RMSprop(alpha)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    return model


def train_model(model, X, Y, epochs):
    return model.fit(X, Y, epochs = epochs, validation_split = 0.2)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # print(hist.columns)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    # plt.ylim([0,20])
    plt.legend()
    plt.show()
