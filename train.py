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
from model import RelationalMemory, FFN
from data import load_data
from utils import calc_actions, calc_reward, calc_metrics
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import copy
import torch

start_time = time.time()

# data params
latency = 0
window_size = 3 * 14
k = 7
sequence_length = 60*8
# sequence_length = 2001 - window_size + 1 - latency - k + 1
commissions = 0#.00075

input_size = 7

mem_slots = 1
num_heads = 4
head_size = 4
num_blocks = 4
# decay_per_layer = 0.6

# NN model definition
model = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=input_size, num_heads=num_heads, num_blocks=num_blocks, forget_bias=1., input_bias=0.)
# model = FFN(input_size, decay_per_layer = decay_per_layer)

# reward function definition
def get_reward(model, state_dict, X, flag_calc_metrics = False, commissions = commissions):
    model.load_state_dict(state_dict)

    initial_capital = 1000
    wealths, buy_amounts, sell_amounts = calc_actions(model, X, sequence_length, latency, window_size, k, initial_capital, commissions)

    reward = calc_reward(wealths, buy_amounts, initial_capital)

    metrics = {}
    if flag_calc_metrics:
        metrics = calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital, X[-sequence_length, 0], X[-1, 0])

    return reward, metrics


def run(start_run, tot_runs, num_iterations, print_steps, output_results, num_workers, save_state_dict, run_evaluation, flag_load_state_dict):
    runs = {}

    hyperparam_search = False
    if (start_run>0 and tot_runs>1): hyperparam_search = True

    # TODO: reset model state_dict
    for i in range(start_run, tot_runs):

        chosen_before = False
        if hyperparam_search:
            npop = np.random.random_integers(num_workers, (160 // num_workers) * num_workers, 1)[0]
            sample = np.random.rand(np.maximum(0,npop))
            sample_std = np.std(sample)
            sigma = np.round(np.sqrt(np.random.chisquare(sample_std,1)),2)[0]
            while sigma == 0:
                sigma = np.round(np.sqrt(np.random.chisquare(sample_std,1)),2)[0]
            learning_rate_selection = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            alpha = np.random.choice(learning_rate_selection)

            for key in runs.keys():
                if runs[key] == [npop, sigma, alpha]:
                    chosen_before = True
                    print('skipping run, as hyperparams [{}] have been chosen before'.format(runs[key]))

        else: #default - best hyperparams
            npop = 52
            sigma = 0.1
            alpha = 0.001

        # will only run if hyperparams are not chosen before
        if not chosen_before:
            runs[i] = [npop, sigma, alpha]

            print('hyperparams chosen -> npop:{} sigma:{} alpha:{}'.format(npop, sigma, alpha))

            if flag_load_state_dict:
                try:
                    model.load_state_dict(torch.load('models/model_state_dict1.pt'))
                except:
                    pass

            es = EvolutionStrategy(copy.deepcopy(model.state_dict()), get_reward, population_size=npop // num_workers,
                                   sigma=sigma,
                                   learning_rate=alpha)

            # train_files, test_files = train_test_split(glob.glob('data/ETH/*.json'), random_state=1)
            train_files = test_files = glob.glob('data/ETH/*.json')[:1]
            print(train_files, test_files)

            if num_workers == 1:
                # single thread version
                metrics = es.run(num_iterations, train_files, model, print_steps, sequence_length, latency, window_size, k)
            else:
                models = []
                for i in range(num_workers):
                    models.append(copy.deepcopy(model))
                # distributed version
                metrics = es.run_dist(num_iterations, train_files, models, print_steps, num_workers, sequence_length, latency, window_size, k)

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
                                                     'benchmark_profit']))

                filename = os.path.join(RUN_SUMMARY_LOC, results['run_name'][0] + str(time.time()) + '.csv')
                results.to_csv(filename, sep=',')

    print("Total Time usage: " + str(timedelta(seconds=int(round(time.time() - start_time)))))

    if save_state_dict:
        # filename = 'models/model_state_dict_' + str(time.time()) + '.pt'
        filename = 'models/model_state_dict.pt'
        print('Saving state_dict to', filename)
        model.load_state_dict(es.state_dict)
        torch.save(model.state_dict(), filename)

        if run_evaluation:
            evaluate(test_files, filename, None, input_size, window_size, mem_slots = mem_slots, num_heads = num_heads, head_size = head_size, num_blocks = num_blocks)



if __name__ == '__main__':
    run(start_run=0, tot_runs=1, num_iterations=20, print_steps=1,
       output_results=True, num_workers=4, save_state_dict=True, run_evaluation=True,
       flag_load_state_dict=False)

    ### hyperparam search
    # run(start_run=1, tot_runs=10, num_iterations=10, print_steps=1,
    #    output_results=True, num_workers=4, save_state_dict=False, run_evaluation=False,
    #    flag_load_state_dict=True)
