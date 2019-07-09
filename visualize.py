import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
from keys import binance_api_key, binance_secret_key
import binance
from binance.client import Client
from binance.enums import *

def main(from_local = True):
    if from_local:
        times = []
        capitals = []
        bnbs = []

        for file in glob.glob('trading_logs/*.json'):
            with open(file, 'r') as f:
                obj = json.load(f)
                times.append(int(obj['initial_time']))
                times.append(int(obj['final_time']))
                capitals.append(float(obj['initial_capital']))
                capitals.append(float(obj['final_capital']))
                bnbs.append(float(obj['initial_bnb']))
                bnbs.append(float(obj['final_bnb']))

        times = np.array(times)
        capitals = np.array(capitals)
        bnbs = np.array(bnbs)

        i = np.argsort(times)
        times = times[i]
        capitals = capitals[i]
        bnbs = bnbs[i]

        times = (times - times[0]) / 60
        capitals = capitals / capitals[0]
        bnbs = bnbs / bnbs[0]

        plt.plot(times, capitals)
        plt.plot(times, bnbs)
        plt.show()
    else:
        client = Client(binance_api_key, binance_secret_key)
        symbol1 = 'ETH'
        symbol = symbol1 + 'USDT'
        trades = client.get_my_trades(symbol=symbol, fromId = 83689922)
        # print(trades[:3])
        df = pd.DataFrame(trades)

        df = df[df['isBuyer'] == False]

        df['quoteQty'] = df['quoteQty'].apply(float)
        df['price'] = df['price'].apply(float)
        df['time'] = df['time'].apply(int)

        prices = df.groupby('time')['price'].mean()
        wealths = df.groupby('time')['quoteQty'].sum()
        times = (df['time'].unique() - df['time'].iloc[0]) / (1000 * 60 * 60 * 24)

        prices = prices / prices.iloc[0]
        wealths = wealths / wealths.iloc[0]

        print('Profit:', wealths.iloc[-1])
        print('min, max:', np.min(wealths), np.max(wealths))

        plt.plot(times, prices)
        plt.plot(times, wealths)
        plt.xlabel('Days')
        plt.ylabel('Profit')
        plt.show()


if __name__ == "__main__":
    main(False)
