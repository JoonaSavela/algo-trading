import matplotlib.pyplot as plt
from utils import plot_y
from model import build_model, train_model, plot_history
from data import get_and_save_data_from_period, load_and_transform_data
import numpy as np
import json
import requests

def main():

    '''
    TODO: data saving structure:
        each coin has its own subdirectory
        each time period of 2000 minutes has its own file; files are spread at least
        2000 minutes apart
    '''

    input_length = 4 * 14

    url = 'https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=2000&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'

    request = requests.get(url)
    timeFrom = 0

    for i in range(3):
        content = json.loads(request._content)
        # print(content)
        print(len(content['Data']))
        print(content.keys())

        for key, item in content.items():
            if key != 'Data':
                print(key, item)

        print(timeFrom - content['TimeTo'])
        print()

        timeFrom = content['TimeFrom']
        url = 'https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=2000&toTs=' + str(timeFrom) + '&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'

        print(timeFrom - content['TimeTo']) # time difference of 2000 minutes in seconds: 120000
        print()

        request = requests.get(url)



    # closes = np.zeros(len(content['Data']))
    # times = np.zeros(len(content['Data']))
    #
    # for i in range(len(content['Data'])):
    #     item = content['Data'][i]
    #     closes[i] = float(item["close"])
    #     times[i] = int(item["time"])
    #
    #
    # plt.plot(times, closes)
    # plt.show()
    #
    # print(times[1] - times[0])

    # X, Y = load_and_transform_data()

    # get_and_save_data_from_period(days = 30)

    # obj = load_data()
    # # print(obj.keys())
    # for v in obj.values():
    #     print(v.keys())
    #     for vv in v.values():
    #         print(vv.keys())
    #         break
    #     break

    # print(X.shape)
    # print(Y.shape)

    # plt.plot(range(len(closes)), closes)
    # plt.show()

    # plot_y(X, Y, input_length * 4 - 1) # last of closes

    # plt.hist(Y)
    # plt.show()

    # model = build_model(input_length * 5, 0.001)
    # # model.summary()
    # history = train_model(model, X, Y, 3)
    # plot_history(history)

    # i = 10
    # tmp = model.predict(X_tmp[0:(i+1),])
    # print(tmp)

if __name__ == "__main__":
    main()
