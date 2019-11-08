# Standard libraries imports
import argparse
import codecs
import datetime
from enum import Enum
from itertools import repeat
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool, Queue, Manager, Lock, Value
import multiprocessing
import numpy as np
import os
import random
import time
import yaml
import sys

from pydrake.math import (RollPitchYaw, RigidTransform)

# Optimization library imports
import nevergrad as ng
import rbfopt
from scipy import optimize
import cma

# Local imports
from ..optimization.eval_parallel import EvalParallel3
from ..optimization.mug_pipeline import MugPipeline, FoundCounterexample, FoundMaxCounterexamples

import torch


class Optimizer():
    highest_process_num = 20

    def __init__(self, num_mugs, mug_initial_pose, mug_lower_bound, mug_upper_bound,
            max_iterations, max_time, max_counterexamples):
        self.mug_initial_poses = []
        for _ in range(num_mugs):
            self.mug_initial_poses += mug_initial_pose

        self.mug_lower_bounds = []
        for _ in range(num_mugs):
            self.mug_lower_bounds += mug_lower_bound

        self.mug_upper_bounds = []
        for _ in range(num_mugs):
            self.mug_upper_bounds += mug_upper_bound

        self.mug_pipeline = MugPipeline(num_mugs=num_mugs, initial_poses=self.mug_initial_poses,
            max_counterexamples=max_counterexamples)
        self.num_mugs = num_mugs

        # Exit conditions
        self.max_time = 10**8
        self.max_counterexamples = 10**10
        self.max_iterations = 10**10

        if max_iterations is not None:
            self.max_iterations = max_iterations

        if max_time is not None:
            self.max_time = max_time         # [s]

        if max_counterexamples is not None:
            self.max_counterexamples = max_counterexamples

        self.num_vars = 7 * self.num_mugs
        self.num_processes = Optimizer.highest_process_num

        self.all_probabilities = None

        # TODO make this global
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 32) & (2**32 - 1))
        random.seed(os.urandom(4))

    @staticmethod
    def listener(q, filename):
        with open(filename, 'w') as f:
            f.write('process_num, iter_num, probability\n')
            while 1:
                m = q.get()
                if m == 'kill':
                    break
                f.write(str(m) + '\n')
                f.flush()

    @staticmethod
    def run_inference(poses, iteration_num, mug_pipeline, all_probabilities, total_iterations,
        num_counterexamples, counter_lock, file_q):
        """
        Wrapper for optimizer's entry point function
        """
        prob = mug_pipeline.run_inference(poses, iteration_num, all_probabilities, total_iterations,
            num_counterexamples, counter_lock, file_q)
        return prob

    # def run_optimizer():
    #     if optimizer_type == 'nevergrad':
    #         plot_graphs(run_nevergrad())
    #     else:
    #         raise Exception('optimizer type not defined')

    def plot_graphs(self):
        fig2, ax = plt.subplots()
        print(self.all_probabilities)
        ax.plot(self.all_probabilities)
        ax.set(xlabel='Iteration', ylabel='Probability')
        ax.set_xlim(xmin=0, xmax=len(self.all_probabilities))
        # ax.set_ylim(ymin=min(self.all_probabilities), ymax=1.0)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.grid()
        fig2.savefig(os.path.join(self.package_directory, 'probability_plot.png'))
        plt.show()

    # def run_nevergrad():
    #     """
        
    #     """
    #     mug_pipeline.set_folder_name("data_nevergrad")

    #     initial_poses = ng.var.Array(1, 7).bounded(-0.5, 0.5)
    #     instrum = ng.Instrumentation(poses=initial_poses)
    #     optimizer = ng.optimizers.RandomSearch(instrumentation=instrum, budget=100)
    #     probability = optimizer.minimize(run_inference)
    #     print(probability)
    #     return all_poses, all_probabilities

    def run_rbfopt(self):
        """
        Rbfopt
        """
        self.mug_pipeline.set_folder_name("data_rbfopt")

        bb = rbfopt.RbfoptUserBlackBox(
            self.num_vars, 
            self.mug_lower_bounds, self.mug_upper_bounds,
            np.array(['R'] * self.num_vars), self.mug_pipeline.run_inference)
        settings = rbfopt.RbfoptSettings(max_evaluations=self.max_iterations)
        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        objval, x, itercount, evalcount, fast_evalcount = alg.optimize()
        state_path = os.path.join(self.package_directory, 'state.dat')
        # print(state_path)
        alg.save_to_file(state_path)

        print('all_poses: {}, all_probabilities: {}'.format(
            self.mug_pipeline.get_all_poses(), self.mug_pipeline.get_all_probabilities()))

    def run_pycma(self):
        """
            Covariance Matrix Evolution Strategy (CMA-ES)
            Note that the bounds have to be rescaled later, for pycma only.
        """

        # self.mug_pipeline.set_folder_name("data_pycma")
        # es = cma.CMAEvolutionStrategy(self.mug_initial_poses, 1.0/3.0, 
        #     {'bounds': [-1.0, 1.0], 'verb_disp': 1})
        # es.optimize(self.mug_pipeline.run_inference)
        # es.result_pretty()

        # cma.plot()

        folder_name = "data_pycma"
        self.mug_pipeline.set_folder_name(folder_name)

        self.mug_initial_poses = []

        for i in range(self.num_mugs):
            self.mug_initial_poses += \
                RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        print(self.mug_initial_poses)

        es = cma.CMAEvolutionStrategy(self.mug_initial_poses, 1.0/3.0,
            {'bounds': [-1.0, 1.0], 'verb_disp': 1})

        iter_num = 0

        start_time = time.time()
        elapsed_time = 0

        manager = Manager()
        self.all_probabilities = manager.list()
        self.total_iterations = manager.Value('d', 0)
        self.num_counterexamples = manager.Value('d', 0)
        counter_lock = manager.Lock()

        file_q = manager.Queue()

        filename = 'robust_perception/optimization/{}/results.csv'.format(folder_name)
        watcher = Process(target=self.listener, args=(file_q, filename))
        watcher.start()

        while not es.stop():
            try:
                ep = EvalParallel3(self.run_inference, number_of_processes=self.num_processes)
                lst = range(iter_num, iter_num + self.num_processes)
                X = es.ask()
                elapsed_time = time.time() - start_time
                ep(X, lst=lst, args=(self.mug_pipeline, self.all_probabilities,
                    self.total_iterations, self.num_counterexamples, counter_lock, file_q),
                    timeout=(self.max_time - elapsed_time))
            except multiprocessing.context.TimeoutError:
                print('timed out!')
                break
            except FoundMaxCounterexamples:
                print('found max counterexamples!')

            iter_num += self.num_processes
            torch.cuda.empty_cache()
            ep.terminate()

        print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
            elapsed_time/60.0, iter_num, elapsed_time/iter_num))
        print('probabilities:', self.all_probabilities)
        es.result_pretty()

        sys.stdout.flush()

    def run_scipy_fmin_slsqp(self):
        """
            Local optimization (sequential least square programming)
        """
        self.mug_pipeline.set_folder_name("data_scipy_fmin_slsqp")

        mug_initial_poses = []
        for i in range(self.num_mugs):
            mug_initial_poses += \
                RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        mug_bound = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (-0.1, 0.1), (0.1, 0.2)]

        mug_bounds = []
        for i in range(0, 3):
            mug_bounds += mug_bound

        exit_mode = optimize.fmin_slsqp(self.mug_pipeline.run_inference, mug_initial_poses,
            bounds=mug_bounds, full_output=True, iter=self.max_iterations)

        print(exit_mode)

    @staticmethod
    def run_nelder_mead_process(process_num, mug_pipeline, num_mugs, all_probabilities,
            total_iterations, total_iterations_lock, file_q):
        # Randomly initialize mug
        mug_initial_poses = []
        num_mugs = 3
        for i in range(num_mugs):
            mug_initial_poses += \
                RollPitchYaw(np.random.uniform(0.0, 2.0*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        mug_lower_bound = (-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1)
        mug_upper_bound = (1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2)

        mug_lower_bounds = []
        mug_upper_bounds = []

        for i in range(0, 3):
            mug_lower_bounds += mug_lower_bound
            mug_upper_bounds += mug_upper_bound

        # print('mug_initial_poses', mug_initial_poses)
        # print('iteration_num', iteration_num)

        while True:
            try:
                # print('process_num', process_num)

                folder = '{}/{}/{:03d}'.format(os.path.dirname(os.path.abspath(__file__)),
                    'data_scipy_nelder_mead', process_num)

                # print('folder', folder)
                if not os.path.exists(folder):
                    os.mkdir(folder)

                optimize.minimize(mug_pipeline.run_inference, mug_initial_poses, 
                    args=(process_num, all_probabilities, total_iterations, total_iterations_lock, file_q),
                    bounds=(mug_lower_bounds, mug_upper_bounds), method='Nelder-Mead',
                    options={'disp': True})
            except FoundCounterexample:
                # After we find a counterex, get new mug initial pose and restart optimizer
                mug_initial_poses = []
                for i in range(num_mugs):
                    mug_initial_poses += \
                        RollPitchYaw(np.random.uniform(0.0, 2.0*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                        [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]
                # TODO put this in a lock
                Optimizer.highest_process_num += 1
                process_num = Optimizer.highest_process_num - 1

    def run_scipy_nelder_mead(self):
        """
        For local search methods, we want to create n parallel processes with random initial poses.
        Each time the process finds a counter example, kill the current process and spawn a
        new process; this allows us to find counter exs that are dissimilar to each other.
        """
        
        start_time = time.time()

        manager = Manager()
        self.all_probabilities = manager.list()
        self.total_iterations = manager.Value('d', 0)
        total_iterations_lock = manager.Lock()

        file_q = manager.Queue()

        folder_name = "data_scipy_nelder_mead"
        self.mug_pipeline.set_folder_name(folder_name)
        pool = Pool(self.num_processes + 1)

        filename = 'robust_perception/optimization/{}/results.csv'.format(folder_name)
        watcher = pool.apply_async(self.listener, (file_q, filename))

        try:
            result = func_timeout(self.max_time, pool.starmap,
                args=(self.run_nelder_mead_process,
                zip(range(self.num_processes), repeat(self.mug_pipeline), repeat(self.num_mugs),
                    repeat(self.all_probabilities), repeat(self.total_iterations), repeat(total_iterations_lock),
                    repeat(file_q))))

        except FunctionTimedOut:
            elapsed_time = time.time() - start_time
            print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
                elapsed_time/60.0, self.total_iterations.value, elapsed_time/self.total_iterations.value))

            file_q.put('kill')
            pool.terminate()
            pool.join()

        end_time = time.time()

        print('probabilities:', self.all_probabilities)

        sys.stdout.flush()

def main():
    ### Initial parameters

    # Parameters that are fairly variable
    optimizer_type = 'pycma'

    max_sec = None
    max_counterexamples = 1
    max_iterations = None
    # max_sec = 60.0 * 60.0 * 5.0
    # max_counterexamples = 100
    # max_iterations = 50000

    # Parameters that are fairly static
    mug_initial_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    optimizer = Optimizer(num_mugs=3, mug_initial_pose=mug_initial_pose,
        mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_iterations=max_iterations, max_time=max_sec,
        max_counterexamples=max_counterexamples)

    multiprocessing.set_start_method('spawn')

    # TODO change to enum
    if optimizer_type == 'pycma':
        print(' in here')
        optimizer.run_pycma()
        optimizer.plot_graphs()
    elif optimizer_type == 'rbfopt':
        optimizer.run_rbfopt()
        optimizer.plot_graphs()
    elif optimizer_type == 'nevergrad':
        optimizer.plot_graphs(optimizer.run_nevergrad())
    elif optimizer_type == 'slsqp':
        optimizer.run_scipy_fmin_slsqp()
        optimizer.plot_graphs()
    elif optimizer_type == 'nelder_mead':
        optimizer.run_scipy_nelder_mead()
        optimizer.plot_graphs()

if __name__ == "__main__":
    main()
