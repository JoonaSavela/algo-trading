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

from keras.models import Model, Input, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam # not important as there's no training here, but required by Keras.
import tensorflow as tf
from keras import backend as K

start_time = time.time()

# data load
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# to run model evluation on 1 core
# config = tf.ConfigProto(intra_op_parallelism_threads=1,
#                         inter_op_parallelism_threads=1,
#                         allow_soft_placement=True, device_count = {'CPU': 1, 'GPU':0})
# session = tf.Session(config=config)
# K.set_session(session)


# data params
batch_size = 60*4
input_length = 4 * 14
# num_classes = 10

# loading data into memory
# x_train = mnist.train.images
# x_val = mnist.validation.images
# x_test = mnist.test.images
# y_train = mnist.train.labels
# y_val = mnist.validation.labels
# y_test = mnist.test.labels


# NN model definition
input_layer = Input(shape=(input_length,6)) # 6 fields in json
flatten = Flatten()(input_layer)
layer_1 = Dense(input_length * 3)(flatten)
layer_2 = Dense(input_length * 3 // 2)(layer_1)
layer_3 = Dense(input_length * 3 // 4)(layer_2)
output_layer = Dense(4, activation='softmax')(layer_3)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse', metrics=['accuracy'])

# reward function definition
def get_reward(weights, calc_metrics = False, latency = 1, random_actions = False):
    filename = np.random.choice(glob.glob('data/*/*.json'))
    # print(filename)
    with open(filename, 'r') as file:
        obj = json.load(file)

    start_index = np.random.choice(len(obj['Data']) - batch_size - input_length - latency)

    data = obj['Data'][start_index:start_index + batch_size + input_length + latency]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key, value in item.items():
            if key != 'time':
                tmp.append(value)
        X[i, :] = tmp

    model.set_weights(weights)

    means = np.reshape(np.mean(X, axis=0), (1,6))
    stds = np.reshape(np.std(X, axis=0), (1,6))

    initial_capital = 1000
    capital_usd = initial_capital
    capital_coin = 0

    wealths = [initial_capital]
    capital_usds = [capital_usd]
    capital_coins = [capital_coin]

    for i in range(batch_size):
        inp = np.reshape((X[i:i+input_length, :] - means) / stds, (1, input_length, 6))
        BUY, SELL, DO_NOTHING, amount = tuple(np.reshape(model.predict(inp), (4,)))
        price = X[i + input_length + latency - 1, 0]

        if BUY > SELL and BUY > DO_NOTHING:
            # print('BUY:', amount, price)
            capital_coin += amount * capital_usd / price
            capital_usd *= 1 - amount
        elif SELL > BUY and SELL > DO_NOTHING:
            # print('SELL:', amount, price)
            capital_usd += amount * capital_coin * price
            capital_coin *= 1 - amount

        wealths.append(capital_usd + capital_coin * price)
        capital_usds.append(capital_usd)
        capital_coins.append(capital_coin)

    price = X[-1, 0]
    # print('SELL:', 1.0, price)
    capital_usd += capital_coin * price
    capital_coin = 0

    wealths.append(capital_usd + capital_coin * price)
    capital_usds.append(capital_usd)
    capital_coins.append(capital_coin)

    wealths = np.array(wealths) / wealths[0] - 1
    # plt.plot(range(batch_size + 2), wealths)
    # plt.show()
    std = np.std(wealths)
    reward = wealths[-1] / (std if std > 0 else 1)
    # print(reward)

    metrics = {}
    if calc_metrics:
        metrics['reward'] = reward
        metrics['profit'] = wealths[-1]
        metrics['max_profit'] = np.max(wealths)
        metrics['min_profit'] = np.min(wealths)
        metrics['std'] = np.std(wealths)
        # metrics['accuracy_train'] = np.mean(np.equal(np.argmax(model.predict(inp),1), np.argmax(solution,1)))

    # print(metrics)

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
                results = pd.DataFrame(np.array(metrics).reshape(int((num_iterations//print_steps)), 8),
                                       columns=list(['run_name',
                                                     'iteration',
                                                     'timestamp',
                                                     'reward',
                                                     'profit',
                                                     'max_profit',
                                                     'min_profit',
                                                     'std']))

                filename = os.path.join(RUN_SUMMARY_LOC, results['run_name'][0] + '.csv')
                results.to_csv(filename, sep=',')

            if save_weights:
                filename = 'models/model_weights.h5'
                print('Saving weights to', filename)
                model.set_weights(es.weights)
                model.save_weights(filename)

    print("Total Time usage: " + str(timedelta(seconds=int(round(time.time() - start_time)))))


if __name__ == '__main__':
    # TODO: Impliment functionality to pass the params via terminal and/or read from config file

    ## single thread run
    run(start_run=0, tot_runs=1, num_iterations=200, print_steps=5,
     output_results=True, num_workers=1, save_weights=True)

    # get_reward(model.get_weights(), calc_metrics=True)

    ### multi worker run
    # run(start_run=0, tot_runs=1, num_iterations=100, print_steps=3,
    #    output_results=False, num_workers=4)

    ### hyperparam search
    #run(start_run=1, tot_runs=100, num_iterations=10000, print_steps=10,
    #    output_results=True, num_workers=1)
