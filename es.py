# dependencies
import numpy as np
import threading
import time
from collections import deque
import glob
from data import load_data
import copy
import torch

# main class for implimenting Evolution Strategies
class EvolutionStrategy(object):

    def __init__(self, model_state_dict, reward_func, population_size, sigma, learning_rate):
        np.random.seed(0)
        self.state_dict = model_state_dict
        self.get_reward = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate

    def get_model_state_dict(self, w, n):
        state_dict_try = copy.deepcopy(w)
        for k, i in n.items():
            jittered = self.SIGMA*i
            state_dict_try[k] += jittered
        return state_dict_try

    # implimention of Algorithm 1: Evolution Strategies by Salimans et al., OpenAI [1], p.2/12
    def run(self, iterations, train_files, model, print_step=10, sequence_length=4*60, latency=0, window_size = 3 * 14, k = 1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE ,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run')

        with torch.no_grad():
            for iteration in range(iterations):

                filename = np.random.choice(train_files)
                X = load_data(filename, sequence_length, latency, window_size, k)

                # checking fitness
                if iteration % print_step == 0:
                    _, return_metrics = self.get_reward(model, self.state_dict, X, flag_calc_metrics=True)
                    print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                    tmp = [run_name, iteration, time.time()]
                    for v in return_metrics.values():
                        tmp.append(v)
                    metrics.append(tmp)

                noise = []
                rewards = torch.zeros(self.POPULATION_SIZE)
                for i in range(self.POPULATION_SIZE):
                    x = {}
                    for key, w in self.state_dict.items():
                        x[key] = torch.randn(w.shape)
                    noise.append(x)

                for i in range(self.POPULATION_SIZE):
                    state_dict_try = self.get_model_state_dict(self.state_dict, noise[i])
                    rewards[i], _  = self.get_reward(model, state_dict_try, X)

                std = torch.std(rewards)
                rewards = (rewards - torch.mean(rewards)) / (std if std != 0 else 1)

                for key, w in self.state_dict.items():
                    A = torch.stack([n[key] for n in noise])

                    # for transposing A
                    permute_indices = list(range(len(A.shape)))[::-1]

                    delta = self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA * (iteration + 1) ** 0.5) * \
                        torch.tensordot(A.permute(*permute_indices), rewards, dims=1)
                    try:
                        delta = delta.permute(*permute_indices[1:])
                    except TypeError:
                        pass
                    self.state_dict[key] += delta

        return metrics

    def worker(self, worker_id, return_container, X, models):
        noise = []
        rewards = torch.zeros(self.POPULATION_SIZE)
        for i in range(self.POPULATION_SIZE):
            x = {}
            for k, w in self.state_dict.items():
                x[k] = torch.randn(w.shape)
            noise.append(x)

        for i in range(self.POPULATION_SIZE):
            state_dict_try = self.get_model_state_dict(self.state_dict, noise[i])
            rewards[i], _  = self.get_reward(models[worker_id], state_dict_try, X)

        std = torch.std(rewards)
        rewards = (rewards - torch.mean(rewards)) / (std if std != 0 else 1)
        return_container[worker_id] = [noise, rewards]

    # Algorithm 2: Parallelized Evolution Strategies by Salimans et al., OpenAI [1], p.3/12
    def run_dist(self, iterations, train_files, models, print_step=10, num_workers=1, sequence_length=4*60, latency=0, window_size = 3 * 14, k = 1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE * num_workers,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run_dist')

        with torch.no_grad():
            for iteration in range(iterations):

                filename = np.random.choice(train_files)
                X = load_data(filename, sequence_length, latency, window_size, k)

                # checking fitness
                if iteration % print_step == 0:
                    _, return_metrics = self.get_reward(models[0], self.state_dict, X, flag_calc_metrics=True)
                    print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                    tmp = [run_name, iteration, time.time()]
                    for v in return_metrics.values():
                        tmp.append(v)
                    metrics.append(tmp)

                return_container = [None] * num_workers
                jobs = []

                for wid in range(0, num_workers):
                    # picking custom seed for each worker
                    # np.random.seed(num_workers * 10)
                    job = threading.Thread(target=self.worker, args=(wid, return_container, X, models))
                    jobs.append(job)
                    job.start()

                for job in jobs:
                    job.join()

                # print(len(return_container))
                noise = []
                rewards = torch.tensor([])

                for worker_output in return_container:
                    noise.extend(worker_output[0])
                    rewards = torch.cat((rewards, worker_output[1]))

                # print(rewards)

                for key, w in self.state_dict.items():
                    A = torch.stack([n[key] for n in noise])

                    # for transposing A
                    permute_indices = list(range(len(A.shape)))[::-1]

                    delta = self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA * (iteration + 1) ** 0.5) * \
                        torch.tensordot(A.permute(*permute_indices), rewards, dims=1)
                    try:
                        delta = delta.permute(*permute_indices[1:])
                    except TypeError:
                        pass
                    self.state_dict[key] += delta

        return metrics
