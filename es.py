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
    def run(self, iterations, train_files, print_step=10, sequence_length=4*60, latency=1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE ,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run')

        for iteration in range(iterations):

            filename = np.random.choice(train_files)
            X = load_data(filename, sequence_length, latency)

            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.state_dict, X, flag_calc_metrics=True)
                print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                tmp = [run_name, iteration, time.time()]
                for v in return_metrics.values():
                    tmp.append(v)
                metrics.append(tmp)

            noise = []
            rewards = torch.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = {}
                for k, w in self.state_dict.items():
                    x[k] = torch.randn(w.shape)
                noise.append(x)

            for i in range(self.POPULATION_SIZE):
                state_dict_try = self.get_model_state_dict(self.state_dict, noise[i])
                rewards[i], _  = self.get_reward(state_dict_try, X)

            std = torch.std(rewards)
            rewards = (rewards - torch.mean(rewards)) / (std if std != 0 else 1)

            for k, w in self.state_dict.items():
                # print(k, noise[0][k])
                A = torch.stack([n[k] for n in noise])
                # print(A.shape, rewards.shape, self.state_dict[k].shape)
                # test = np.array([n[k].data.numpy() for n in noise])
                # print(test.shape, test.T.shape, np.dot(test.T, rewards).T.shape)
                
                # for transposing A
                permute_indices = list(range(len(A.shape)))[::-1]

                delta = self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * \
                    torch.tensordot(A.permute(*permute_indices), rewards, dims=1)
                try:
                    delta = delta.permute(*permute_indices[1:])
                except TypeError:
                    pass
                self.state_dict[k] += delta
        return metrics

    def worker(self, worker_id, return_container, X):
        noise = []
        rewards = np.zeros(self.POPULATION_SIZE)

        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.state_dict:
                x.append(np.random.randn(*w.shape))
            noise.append(x)

        for i in range(self.POPULATION_SIZE):
            state_dict_try = self.get_model_state_dict(self.state_dict, noise[i])
            rewards[i], _  = self.get_reward(state_dict_try, X)

        std = np.std(rewards)
        rewards = (rewards - np.mean(rewards)) / (std if std != 0 else 1)
        return_container[worker_id] = [noise, rewards]

    # Algorithm 2: Parallelized Evolution Strategies by Salimans et al., OpenAI [1], p.3/12
    # TODO: update like the single thread version
    def run_dist(self, iterations, train_files, print_step=10, num_workers=1, sequence_length=4*60, latency=1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE * num_workers,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run_dist')
        for iteration in range(iterations):

            filename = np.random.choice(train_files)
            X = load_data(filename, sequence_length, latency)

            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.state_dict, X, flag_calc_metrics=True)
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
                job = threading.Thread(target=self.worker, args=(wid, return_container, X))
                jobs.append(job)
                job.start()

            for job in jobs:
                job.join()

            # print(len(return_container))
            noise = []
            rewards = []

            for worker_output in return_container:
                noise.extend(worker_output[0])
                rewards.extend(worker_output[1])

            for index, w in enumerate(self.state_dict):
                A = np.array([n[index] for n in noise])
                self.state_dict[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T

        return metrics
