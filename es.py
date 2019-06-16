# dependencies
import numpy as np
import threading
import time
from collections import deque
import glob
from data import load_data

# main class for implimenting Evolution Strategies
class EvolutionStrategy(object):

    def __init__(self, model_weights, reward_func, population_size, sigma, learning_rate):
        np.random.seed(0)
        self.weights = model_weights
        self.get_reward = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate

    def get_model_weights(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try

    # implimention of Algorithm 1: Evolution Strategies by Salimans et al., OpenAI [1], p.2/12
    def run(self, iterations, train_files, print_step=10, batch_size=4*60, input_length=4*14, latency=1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE ,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run')

        for iteration in range(iterations):

            filename = np.random.choice(train_files)
            X = load_data(filename, batch_size, input_length, latency)

            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.weights, X, flag_calc_metrics=True)
                print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                tmp = [run_name, iteration, time.time()]
                for v in return_metrics.values():
                    tmp.append(v)
                metrics.append(tmp)

            noise = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                noise.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self.get_model_weights(self.weights, noise[i])
                rewards[i], _  = self.get_reward(weights_try, X)

            std = np.std(rewards)
            rewards = (rewards - np.mean(rewards)) / (std if std != 0 else 1)

            for index, w in enumerate(self.weights):
                A = np.array([n[index] for n in noise])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
        return metrics

    def worker(self, worker_id, return_container, X):
        noise = []
        rewards = np.zeros(self.POPULATION_SIZE)

        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            noise.append(x)

        for i in range(self.POPULATION_SIZE):
            weights_try = self.get_model_weights(self.weights, noise[i])
            rewards[i], _  = self.get_reward(weights_try, X)

        std = np.std(rewards)
        rewards = (rewards - np.mean(rewards)) / (std if std != 0 else 1)
        return_container[worker_id] = [noise, rewards]

    # Algorithm 2: Parallelized Evolution Strategies by Salimans et al., OpenAI [1], p.3/12
    # TODO: update like the single thread version
    def run_dist(self, iterations, train_files, print_step=10, num_workers=1, batch_size=4*60, input_length=4*14, latency=1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE * num_workers,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run_dist')
        for iteration in range(iterations):

            filename = np.random.choice(train_files)
            X = load_data(filename, batch_size, input_length, latency)

            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.weights, X, flag_calc_metrics=True)
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

            for index, w in enumerate(self.weights):
                A = np.array([n[index] for n in noise])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T

        return metrics
