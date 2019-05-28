import matplotlib.pyplot as plt
from utils import plot_y
from model import build_model, train_model, plot_history
from data import get_and_save_data_from_period
import numpy as np
import json

def main():

    input_length = 4 * 14

    get_and_save_data_from_period()

    # a = np.arange(10).reshape(2,5) # a 2 by 5 array
    # b = a.tolist() # nested lists with same data, indices
    # print(b)
    # print(np.array(b))

    # with open('data/test.json', 'w') as testfile:
    #     json.dump({'test': 3}, testfile)

    # print(X.shape)
    # print(Y.shape[1])

    # plt.plot(range(len(closes)), closes)
    # plt.show()

    # plot_y(X, Y, input_length * 4 - 1) # last of closes

    # plt.hist(Y)
    # plt.show()

    # model = build_model(input_length * 5, 0.001)
    # # model.summary()
    # history = train_model(model, X, Y, 10)
    # plot_history(history)

    # i = 10
    # tmp = model.predict(X_tmp[0:(i+1),])
    # print(tmp)

if __name__ == "__main__":
    main()
