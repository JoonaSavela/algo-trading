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
from utils import calc_actions, calc_reward, calc_metrics
from sklearn.model_selection import train_test_split
from evaluate import evaluate

from keras.models import Model, Input, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam # not important as there's no training here, but required by Keras.
import tensorflow as tf
from keras import backend as K

start_time = time.time()

# data params
batch_size = 60*4
input_length = 4 * 14
latency = 0
# num_classes = 10

# NN model definition
model = build_model()

# reward function definition
def get_reward(weights, X, flag_calc_metrics = False, random_actions = False, commissions = 0.00075):
    # filename = np.random.choice(glob.glob('data/*/*.json'))
    # X = load_data(filename, batch_size, input_length, latency)
    price_max = np.max(X[input_length:input_length+batch_size,0]) / X[input_length, 0]
    price_min = np.min(X[input_length:input_length+batch_size,0]) / X[input_length, 0]

    model.set_weights(weights)

    initial_capital = 1000
    wealths, buy_amounts, sell_amounts = calc_actions(model, X, batch_size, input_length, latency, initial_capital, commissions)

    reward = calc_reward(wealths)

    metrics = {}
    if flag_calc_metrics:
        metrics = calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital)

    return reward, metrics


def run(start_run, tot_runs, num_iterations, print_steps, output_results, num_workers, save_weights, run_evaluation):
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

            train_files, test_files = train_test_split(glob.glob('data/*/*.json'))

            if num_workers == 1:
                # single thread version
                metrics = es.run(num_iterations, train_files, print_steps, batch_size, input_length, latency)
            else:
                # distributed version
                es.run_dist(num_iterations, print_steps, num_workers)

            if output_results:
                RUN_SUMMARY_LOC = './run_summaries/'
                print('saving results to loc:', RUN_SUMMARY_LOC )
                # TODO: make reshaping better
                results = pd.DataFrame(np.array(metrics).reshape(-1, len(metrics[0])),
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

                if run_evaluation:
                    evaluate(test_files, filename)

    print("Total Time usage: " + str(timedelta(seconds=int(round(time.time() - start_time)))))


if __name__ == '__main__':
    # TODO: Impliment functionality to pass the params via terminal and/or read from config file

    ## single thread run
    run(start_run=0, tot_runs=1, num_iterations=350, print_steps=10,
     output_results=True, num_workers=1, save_weights=True, run_evaluation=True)

    ### multi worker run
    # run(start_run=0, tot_runs=1, num_iterations=100, print_steps=3,
    #    output_results=False, num_workers=4)

    ### hyperparam search
    #run(start_run=1, tot_runs=100, num_iterations=10000, print_steps=10,
    #    output_results=True, num_workers=1)
