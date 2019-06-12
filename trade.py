from keys import binance_api_key, binance_secret_key
from model import build_model
import binance
from binance.client import Client
import time
from data import get_recent_data
import numpy as np

weights_filename = 'models/model_weights.h5'
input_length = 14 * 4

def order(client):
    pass

def cancel_orders(client):
    pass



def trading_pipeline():
    model = build_model()
    model.load_weights(weights_filename)

    client = Client(binance_api_key, binance_secret_key)

    try:
        print()
        print('Starting trading pipeline...')
        # print(client.get_all_orders(symbol='BTCUSDT'))
        # print(X.shape)
        while True:
            print('Fetching data')
            X, timeTo = get_recent_data('BTC')
            print(timeTo, model.predict( np.reshape(X, (1, input_length, 6)) ))
            time_diff = time.time() - timeTo
            waiting_time = 60 - time_diff
            print('waiting', waiting_time, 'seconds')
            time.sleep(waiting_time)
            print()
    except KeyboardInterrupt:
        print()
        print('Exiting trading pipeline...')
    finally:
        cancel_orders(client)


    return


if __name__ == '__main__':
    trading_pipeline()
