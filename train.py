# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
from es import EvolutionStrategy
import matplotlib.pyplot as plt
from model import build_model
from data import load_data

from keras.models import Model, Input, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam # not important as there's no training here, but required by Keras.
import tensorflow as tf
from keras import backend as K

start_time = time.time()

# data params
batch_size = 60*4
input_length = 4 * 14
# num_classes = 10

# NN model definition
model = build_model()

# reward function definition
def get_reward(weights, calc_metrics = False, latency = 1, random_actions = False, commissions = 0.00075):
    filename = np.random.choice(glob.glob('data/*/*.json'))
    X = load_data(filename, batch_size, input_length, latency)

    means = np.reshape(np.mean(X, axis=0), (1,6))
    stds = np.reshape(np.std(X, axis=0), (1,6))

    initial_capital = 1000
    capital_usd = initial_capital
    capital_coin = 0

    wealths = [initial_capital]
    capital_usds = [capital_usd]
    capital_coins = [capital_coin]
    buy_amounts = []
    sell_amounts = []

    for i in range(batch_size):
        inp = np.reshape((X[i:i+input_length, :] - means) / stds, (1, input_length, 6))
        BUY, SELL, DO_NOTHING, amount = tuple(np.reshape(model.predict(inp), (4,)))
        price = X[i + input_length + latency - 1, 0]

        if BUY > SELL and BUY > DO_NOTHING:
            amount_coin = amount * capital_usd / price * (1 - commissions)
            capital_coin += amount_coin
            capital_usd *= 1 - amount
            buy_amounts.append(amount_coin * price)
        elif SELL > BUY and SELL > DO_NOTHING:
            amount_usd = amount * capital_coin * price * (1 - commissions)
            capital_usd += amount_usd
            capital_coin *= 1 - amount
            sell_amounts.append(amount_usd)

        wealths.append(capital_usd + capital_coin * price)
        capital_usds.append(capital_usd)
        capital_coins.append(capital_coin)

    price = X[-1, 0]
    capital_usd += capital_coin * price
    capital_coin = 0

    wealths.append(capital_usd + capital_coin * price)
    capital_usds.append(capital_usd)
    capital_coins.append(capital_coin)
    sell_amounts.append(1)

    wealths = np.array(wealths) / wealths[0] - 1
    std = np.std(wealths)
    reward = wealths[-1] / (std if std > 0 else 1)
    # reward = wealths[-1] / (np.sum(buy_amounts) + np.sum(sell_amounts)) * initial_capital

    metrics = {}
    if calc_metrics:
        metrics['reward'] = reward
        metrics['profit'] = wealths[-1]
        metrics['max_profit'] = np.max(wealths)
        metrics['min_profit'] = np.min(wealths)
        metrics['buys'] = np.sum(buy_amounts) / initial_capital
        metrics['sells'] = np.sum(sell_amounts) / initial_capital
        metrics['std'] = np.std(wealths)

    return reward, metrics


def run(start_run, tot_runs, num_iterations, print_steps, output_results, num_workers, save_weights):
    runs = {}

    hyperparam_search = False
    if (start_run>0 and tot_runs>1): hyperparam_search = True


    for i in range(start_run, tot_runs):

        chosen_before = False
        if hyperparam_search:
            npop = np.random.random_integers(1, 150, 1)[0]
            sample = np.random.rand(np.maximum(0,npop))
            sample_std = np.std(sample)
            sigma = np.round(np.sqrt(np.random.chisquare(sample_std,1)),2)[0]
            learning_rate_selection = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            alpha = np.random.choice(learning_rate_selection)

            for key in runs.keys():
                if runs[key] == [npop, sigma, alpha]:
                    chosen_before = True
                    print('skipping run, as hyperparams [{}] have been chosen before'.format(hyperparams))

        else: #default - best hyperparams
            npop = 50
            sigma = 0.1
            alpha = 0.001

        # will only run if hyperparams are not chosen before
        if not chosen_before:
            runs[i] = [npop, sigma, alpha]

            print('hyperparams chosen -> npop:{}  sigma:{} alpha:{}'.format(npop, sigma, alpha))

            es = EvolutionStrategy(model.get_weights(), get_reward, population_size=npop,
                                   sigma=sigma,
                                   learning_rate=alpha)

            if num_workers == 1:
                # single thread version
                metrics = es.run(num_iterations, print_steps)
            else:
                # distributed version
                es.run_dist(num_iterations, print_steps, num_workers)

            if output_results:
                RUN_SUMMARY_LOC = './run_summaries/'
                print('saving results to loc:', RUN_SUMMARY_LOC )
                results = pd.DataFrame(np.array(metrics).reshape(int((num_iterations//print_steps)), len(metrics[0])),
                                       columns=list(['run_name',
                                                     'iteration',
                                                     'timestamp',
                                                     'reward',
                                                     'profit',
                                                     'max_profit',
                                                     'min_profit',
                                                     'buys',
                                                     'sells',
                                                     'std']))

                filename = os.path.join(RUN_SUMMARY_LOC, results['run_name'][0] + str(time.time()) + '.csv')
                results.to_csv(filename, sep=',')

            if save_weights:
                filename = 'models/model_weights_' + str(time.time()) + '.h5'
                print('Saving weights to', filename)
                model.set_weights(es.weights)
                model.save_weights(filename)

    print("Total Time usage: " + str(timedelta(seconds=int(round(time.time() - start_time)))))


if __name__ == '__main__':
    # TODO: Impliment functionality to pass the params via terminal and/or read from config file

    ## single thread run
    run(start_run=0, tot_runs=1, num_iterations=250, print_steps=5,
     output_results=True, num_workers=1, save_weights=True)

    # get_reward(model.get_weights(), calc_metrics=True)

    ### multi worker run
    # run(start_run=0, tot_runs=1, num_iterations=100, print_steps=3,
    #    output_results=False, num_workers=4)

    ### hyperparam search
    #run(start_run=1, tot_runs=100, num_iterations=10000, print_steps=10,
    #    output_results=True, num_workers=1)
