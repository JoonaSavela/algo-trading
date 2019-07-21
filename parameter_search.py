from strategy import *
import numpy as np
import pandas as pd
import glob
from utils import get_time
import os

def random_search(files, n_runs, strategy_class, stop_loss_take_profit, restrictive):
    X = load_all_data(files)

    options = strategy_class.get_options(stop_loss_take_profit, restrictive)
    #print('options:', options)

    if not os.path.exists('search_results/'):
        os.mkdir('search_results/')

    filename = 'search_results/random_search_{}_{}_{}.csv'.format(
        options['name'],
        stop_loss_take_profit,
        restrictive
    )
    print(filename)

    try:
        resulting_df = pd.read_csv(filename, index_col = 0)
    except FileNotFoundError as e:
        print(e)
        resulting_df = pd.DataFrame()
    print(resulting_df)

    for run in range(n_runs):
        try:
            chosen_params = dict()

            for k, v in options.items():
                if k != 'name':
                    type, args = v
                    if type == 'range':
                        chosen_params[k] = int(np.random.choice(np.arange(*args)))
                    elif type == 'uniform':
                        chosen_params[k] = float(np.random.uniform(*args))

            strategy = strategy_class(chosen_params, stop_loss_take_profit, restrictive)

            profit, min_profit, max_profit = evaluate_strategy(X, strategy, False)

            chosen_params['profit'] = float(profit)
            chosen_params['min_profit'] = float(min_profit)
            chosen_params['max_profit'] = float(max_profit)

            ser = pd.DataFrame({len(resulting_df): chosen_params}).T
            resulting_df = pd.concat([resulting_df, ser], ignore_index=True)

            print()
            print(resulting_df.loc[resulting_df.index[-1]:, ['profit', 'max_profit', 'min_profit']])
            resulting_df.to_csv(filename)
        except ValueError as e:
            print(e)

    print(resulting_df.loc[resulting_df['profit'].idxmax(), :])

if __name__ == '__main__':
    n_runs = 1000
    strategy_class = Main_Strategy
    stop_loss_take_profit = True
    restrictive = True

    dir = 'data/ETH/'
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)
    print(dir, len(files), round(len(files) * 2001 / (60 * 24)))

    random_search(files, n_runs, strategy_class, stop_loss_take_profit, restrictive)
