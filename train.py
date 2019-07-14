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
from utils import stochastic_oscillator, heikin_ashi, sma, std, get_time, round_to_n
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import copy
import torch
import torch.nn as nn
from optimize import get_wealths

# reward function definition
def get_reward(model, state_dict, X, flag_calc_metrics = False, commissions = 0.00075):
    model.load_state_dict(state_dict)

    initial_capital = 1000
    wealths, buy_amounts, sell_amounts = calc_actions(model, X, sequence_length, latency, window_size, k, initial_capital, commissions)

    reward = calc_reward(wealths, buy_amounts, initial_capital)

    metrics = {}
    if flag_calc_metrics:
        metrics = calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital, X[-sequence_length, 0], X[-1, 0])

    return reward, metrics


def run(start_run, tot_runs, num_iterations, print_steps, output_results, num_workers, save_state_dict, run_evaluation, flag_load_state_dict):
    start_time = time.time()
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


def calc_output(model, filename, sequence_length, window_size1, window_size2, window_size3, k, reset_memory = False):
    if reset_memory:
        X, start_index = load_data(
            filename, 2001, 0, \
            window_size1 + np.max([window_size3, window_size2]) - 1, k, True
        )

    else:
        X, start_index = load_data(
            filename, sequence_length, 0, \
            window_size1 + np.max([window_size3, window_size2]) - 1, k, True
        )

    stochastic = stochastic_oscillator(X, window_size2, k)

    X = X[-stochastic.shape[0]:, :]

    if reset_memory:
        i = 0
        buys_list = []
        while i < 2001:
            inp = np.concatenate(
                [X[i:i + sequence_length, :4]],
                axis = 1
            ).astype(np.float32)
            inp = torch.from_numpy(inp).view(1, inp.shape[0], model.input_size)
            inp[:, :4] /= X[i, 0]

            memory = model.initial_state(batch_size = 1)

            buys, memory = model(inp, memory)
            buys_list.append(buys)

            i += sequence_length

        buys = torch.cat(buys_list)
    else:
        inp = torch.from_numpy(
            np.concatenate(
                [X[:, :4]],
                axis = 1
            ).astype(np.float32)
        ).view(1, X.shape[0], model.input_size)
        inp[:, :4] /= X[0, 0]

        memory = model.initial_state(batch_size = 1)

        buys, memory = model(inp, memory)

    return buys, X, start_index


def train(model, n_epochs, lr, n_iter_per_file, sequence_length, window_size1, window_size2, \
          window_size3, commissions, print_step = 10, save_model = False, load_model = False):
    start_time = time.time()

    if load_model:
        model.load_state_dict(torch.load('models/model_state_dict.pt'))
        print('model loaded')

    # train_files, test_files = train_test_split(glob.glob('data/ETH/*.json'), random_state=1)
    train_files = test_files = sorted(glob.glob('data/ETH/*.json'), key = get_time)[:3]
    print(train_files, test_files)

    criterion = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(n_epochs):
        print_i = torch.randint(len(train_files), (1,))
        for file_i, filename in enumerate(train_files):
            for iter in range(n_iter_per_file):
                optim.zero_grad()

                buys, X, start_index = calc_output(model, filename, sequence_length, window_size1, window_size2, window_size3, k)

                t = str(get_time(filename))
                coin = filename.split('/')[1]
                dir = 'data/labels/' + coin + '/'

                df = pd.read_csv(dir + t + '.csv', index_col = 0, header = None)
                start_index += window_size2 - 1 + k - 1
                buys_optim = torch.from_numpy(
                    df.values[start_index:start_index + sequence_length].astype(np.float32)
                )

                loss = criterion(buys, buys_optim)
                loss.backward()
                optim.step()


            if epoch % print_step == 0 and file_i == print_i:
                buys = buys.view(-1).detach().numpy()
                buys_optim = buys_optim.view(-1).detach().numpy()

                wealths, _, _, _, _ = get_wealths(X, buys, commissions = commissions)
                wealths_optim, _, _, _, _ = get_wealths(X, buys_optim, commissions = commissions)

                print(
                    '[Epoch: {}/{}] [Loss: {:.4}] [Profit: {:.4}] [Benchmark: {:.4}] [Optimal: {:.4}]'.format(
                        epoch, n_epochs, loss.item(), wealths[-1], X[-1, 0] / X[0, 0] - 1, wealths_optim[-1]
                    )
                )

    print("Total Time usage: " + str(timedelta(seconds=int(round(time.time() - start_time)))))

    Xs = []
    buys_test = []
    buys_test_optim = []

    for filename in test_files:
        buys, X, start_index = calc_output(model, filename, sequence_length, window_size1, window_size2, window_size3, k, True)
        start_index += window_size2 - 1 + k - 1

        buys_test.append(buys.view(-1).detach().numpy())
        Xs.append(X)

        t = str(get_time(filename))
        coin = filename.split('/')[1]
        dir = 'data/labels/' + coin + '/'

        df = pd.read_csv(dir + t + '.csv', index_col = 0, header = None)
        buys_optim = df.values.reshape(-1)
        buys_test_optim.append(buys_optim[start_index:])

    buys = np.concatenate(buys_test)
    buys_optim = np.concatenate(buys_test_optim)
    X = np.concatenate(Xs)

    wealths, _, _, _, _ = get_wealths(X, buys, commissions = commissions)
    wealths_optim, _, _, _, _ = get_wealths(X, buys_optim, commissions = commissions)

    print(
        '[Test] [Profit: {:.4}] [Benchmark: {:.4}] [Optimal: {:.4}]'.format(
            wealths[-1], X[-1, 0] / X[0, 0] - 1, wealths_optim[-1]
        )
    )

    plt.plot(range(X.shape[0]), X[:, 0] / X[0, 0])
    plt.plot(range(X.shape[0]), wealths + 1)
    plt.plot(range(X.shape[0]), wealths_optim + 1)
    plt.show()

    if save_model:
        torch.save(model.state_dict(), 'models/model_state_dict1.pt')
        print('model saved')

if __name__ == '__main__':
    # data params
    window_size1 = 1#3 * 14
    window_size2 = 1# * 14
    window_size3 = 1# * 14
    window_size = window_size1 + np.max([window_size3, window_size2]) - 1
    k = 1
    sequence_length = 60
    # sequence_length = 2001 - window_size + 1 - k + 1
    commissions = 0.00075

    input_size = 4

    mem_slots = 1
    num_heads = 15
    head_size = 6
    num_blocks = 6
    # decay_per_layer = 0.6

    # NN model definition
    model = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=input_size, num_heads=num_heads, num_blocks=num_blocks, forget_bias=1., input_bias=0.)
    # model = FFN(input_size, decay_per_layer = decay_per_layer)

    n_epochs = 20
    print_step = max(n_epochs // 20, 1)
    lr = 0.001
    n_iter_per_file = 50

    train(
        model = model,
        n_epochs = n_epochs,
        lr = lr,
        n_iter_per_file = n_iter_per_file,
        sequence_length = sequence_length,
        window_size1 = window_size1,
        window_size2 = window_size2,
        window_size3 = window_size3,
        commissions = commissions,
        print_step = print_step,
        save_model = True,
        load_model = True,
    )





    # run(start_run=0, tot_runs=1, nu'm_iterations=20, print_steps=1,
    #    output_results=True, num_workers=4, save_state_dict=True, run_evaluation=True,
    #    flag_load_state_dict=False)'

    ### hyperparam search
    # run(start_run=1, tot_runs=10, num_iterations=10, print_steps=1,
    #    output_results=True, num_workers=4, save_state_dict=False, run_evaluation=False,
    #    flag_load_state_dict=True)
