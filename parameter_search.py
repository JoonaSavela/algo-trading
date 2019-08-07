from strategy import *
import numpy as np
import pandas as pd
import glob
from utils import get_time
import os
from parameters import parameters

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

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

            profit, min_profit, max_profit = evaluate_strategy(X, strategy, 0, False)

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


def optimise(coin, files, strategy_class, stop_loss_take_profit, restrictive, kappa, n_runs):
    X = load_all_data(files)

    if restrictive:
        starts = [0, X.shape[0] // 2]
    else:
        starts = [0]

    n_months = (X.shape[0] * len(starts) - sum(starts)) / (60 * 24 * 30)

    print('Number of months:', n_months)

    def objective_function(stop_loss,
                       decay,
                       take_profit,
                       maxlen,
                       waiting_time,
                       window_size1,
                       window_size2,
                       window_size,
                       look_back_size,
                       rolling_min_window_size,
                       min_threshold,
                       change_threshold,
                       buy_threshold,
                       sell_threshold,
                       ha_threshold,
                       c):

        maxlen = int(maxlen)
        waiting_time = int(waiting_time)
        window_size1 = int(window_size1)
        window_size2 = int(window_size2)
        window_size = int(window_size)
        look_back_size = int(look_back_size)
        rolling_min_window_size = int(rolling_min_window_size)

        params = {
            'stop_loss': stop_loss,
            'decay': decay,
            'take_profit': take_profit,
            'maxlen': maxlen,
            'waiting_time': waiting_time,
            'window_size1': window_size1,
            'window_size2': window_size2,
            'window_size': window_size,
            'look_back_size': look_back_size,
            'rolling_min_window_size': rolling_min_window_size,
            'min_threshold': min_threshold,
            'change_threshold': change_threshold,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'ha_threshold': ha_threshold,
            'c': c,
        }

        strategy = strategy_class(params, stop_loss_take_profit, restrictive)
        if strategy.size() > 2000:
            return -1

        profits = []
        trades_list = []

        for start in starts:
            profit, min_profit, max_profit, trades, biggest_loss = evaluate_strategy(X, strategy, start, 0, False)

            profits.append(profit)
            trades_list.append(trades)

        profit = np.prod(profits) * biggest_loss
        trades = np.concatenate(trades_list)

        if trades.shape[0] == 0:
            return -1

        score = profit ** (1 / n_months) - 1
        if score > 0 and trades.shape[0] > 2:
            score /= np.std(trades)

        print('Profit:', profit ** (1 / n_months), 'Score:', score, 'Biggest loss:', biggest_loss)

        return score


    options = strategy_class.get_options(stop_loss_take_profit, restrictive)

    # Bounded region of parameter space
    pbounds = {}

    for k, v in options.items():
        if k != 'name':
            type, bounds = v
            pbounds[k] = bounds # TODO: set bounds differently if not in strategy class

    # print(pbounds)

    optimizer = BayesianOptimization(
        f = objective_function,
        pbounds = pbounds,
        # random_state = 1,
    )

    filename = 'optim_results/{}_{}_{}_{}.json'.format(
        coin,
        options['name'],
        stop_loss_take_profit,
        restrictive
    )

    # load_logs(optimizer, logs=[filename])

    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    for obj in parameters:
        params = obj['params']
        for k, v in options.items():
            if k != 'name' and k not in params:
                type, bounds = v
                params[k] = bounds[0]

        keys_to_pop = []
        for k, v in params.items():
            if k not in options:
                keys_to_pop.append(k)

        for k in keys_to_pop:
            params.pop(k)

        optimizer.probe(
            params=params,
            lazy=True,
        )

    optimizer.maximize(
        init_points = int(np.sqrt(n_runs)),
        n_iter = n_runs,
    )

    # fix_optim_log(filename)

    print(optimizer.max)




if __name__ == '__main__':
    n_runs = 800
    kappa = 1
    strategy_class = Main_Strategy
    stop_loss_take_profit = True
    restrictive = True

    coin = 'ETH'

    dir = 'data/' + coin + '/'
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)
    # print(dir, len(files), round(len(files) * 2001 / (60 * 24)))

    optimise(coin, files, strategy_class, stop_loss_take_profit, restrictive, kappa, n_runs)
    # random_search(files, n_runs, strategy_class, stop_loss_take_profit, restrictive)
