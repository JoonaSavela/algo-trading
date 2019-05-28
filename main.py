import matplotlib.pyplot as plt
from utils import get_data, normalize_values, transform_data, plot_y
from model import build_model, train_model, plot_history
import numpy as np
import pandas as pd

def main():
    # TODO: move this into a function; save the result into a file
    stocks = pd.read_csv('stocks.csv')

    input_length = 4 * 14
    X = np.empty(shape=[0, input_length * 5]) # 5 = number of types of input variables
    Y = np.empty(shape=[0, 3])

    for stock in stocks['Symbol']:
        try:
            # TODO: all dates from 30 days ago
            url = 'https://api.iextrading.com/1.0/stock/'+stock+'/chart/date/20190523'
            opens, highs, lows, closes, volumes = get_data(url)

            opens, highs, lows, closes, volumes = normalize_values(opens, highs, lows, closes, volumes)

            newX, newY = transform_data(opens, highs, lows, closes, volumes, input_length)

            X = np.append(X, newX, axis = 0)
            Y = np.append(Y, newY, axis = 0)

            print("Appended " + stock + " (" + str(newY.shape[0]) + " rows)" + ".")
        except KeyError:
            pass
        break


    print(X.shape)
    print(Y.shape[1])

    # plt.plot(range(len(closes)), closes)
    # plt.show()

    plot_y(X, Y, input_length * 4 - 1) # last of closes

    # plt.hist(Y)
    # plt.show()

    # model = build_model(input_length * 5, 0.001)
    # # model.summary()
    # history = train_model(model, X, Y, 10)
    # plot_history(history)

    # i = 10
    # tmp = model.predict(X_tmp[0:(i+1),])
    # print(tmp)

    # i = 0
    # print(X.shape)
    # print(Y.shape)
    # print(X[i,(input_length * 4):(input_length * 5)])
    # print(Y[i])

    # print(len(times))
    # plt.plot(times, closes)
    # plt.show()

if __name__ == "__main__":
    main()
