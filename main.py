import matplotlib.pyplot as plt
from utils import plot_y
from model import build_model, train_model, plot_history
from data import get_and_save_data_from_period, load_and_transform_data
import numpy as np
import json

def main():

    input_length = 4 * 14

    X, Y = load_and_transform_data()

    # get_and_save_data_from_period(days = 30)

    # obj = load_data()
    # # print(obj.keys())
    # for v in obj.values():
    #     print(v.keys())
    #     for vv in v.values():
    #         print(vv.keys())
    #         break
    #     break

    print(X.shape)
    print(Y.shape)

    # plt.plot(range(len(closes)), closes)
    # plt.show()

    # plot_y(X, Y, input_length * 4 - 1) # last of closes

    # plt.hist(Y)
    # plt.show()

    model = build_model(input_length * 5, 0.001)
    # model.summary()
    history = train_model(model, X, Y, 3)
    plot_history(history)

    # i = 10
    # tmp = model.predict(X_tmp[0:(i+1),])
    # print(tmp)

if __name__ == "__main__":
    main()
