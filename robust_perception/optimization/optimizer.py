import argparse
import codecs
import datetime
from enum import Enum
from func_timeout import func_timeout, FunctionTimedOut
from itertools import repeat
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

from pydrake.math import (RollPitchYaw, RigidTransform)

# Optimization library imports
import rbfopt
from scipy import optimize
import cma

from torch.multiprocessing import Process, Pool, Queue, Manager, Lock, Value
import torch.multiprocessing

class OptimizerType(Enum):
    """
    Summary of each optimizer type here: bit.ly/TODO
    """
    # Global optimizers
    PYCMA = 1
    RBFOPT = 2

    # Local optimizers
    NELDER_MEAD = 3
    SLSQP = 4

    # Random sample
    RANDOM = 5

# Local imports
from ..optimization.eval_parallel import EvalParallel3
from ..optimization.mug_pipeline import MugPipeline, FoundCounterexample, FoundMaxCounterexamples
from ..optimization.model_trainer import MyNet

import torch

class Optimizer():
    highest_process_num = 0     # Number of reinitializations for local optimizer

    def __init__(self, num_mugs, mug_lower_bound, mug_upper_bound,
            max_iterations, max_time, max_counterexamples, num_processes,
            retrain_with_counterexamples, mug_initial_poses=[]):
        # torch.multiprocessing.set_start_method('spawn')

        self.mug_initial_poses = mug_initial_poses

        self.mug_lower_bounds = []
        for _ in range(num_mugs):
            self.mug_lower_bounds += mug_lower_bound

        self.mug_upper_bounds = []
        for _ in range(num_mugs):
            self.mug_upper_bounds += mug_upper_bound

        self.mug_pipeline = MugPipeline(
            num_mugs=num_mugs, max_counterexamples=max_counterexamples,
            retrain_with_counterexamples=retrain_with_counterexamples)
        self.num_mugs = num_mugs

        # Exit conditions, initialized to some large numbers
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

        global highest_process_num
        highest_process_num = num_processes
        self.num_processes = num_processes

        self.all_probabilities = None

        self.retrain_with_counterexamples = retrain_with_counterexamples

        # TODO make this global
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 32) & (2**32 - 1))
        random.seed(os.urandom(4))

    @staticmethod
    def run_inference(poses, iteration_num, mug_pipeline, all_probabilities, total_iterations,
        num_counterexamples, model_number, model_number_lock, counter_lock, all_probabilities_lock, file_q):
        """
        Wrapper for optimizer's entry point function
        """
        print('at beg of run_inference iter {}'.format(iteration_num), flush=True)

        prob = mug_pipeline.run_inference(
            poses, iteration_num, all_probabilities, total_iterations,
            num_counterexamples, model_number, model_number_lock, counter_lock, all_probabilities_lock, file_q)

        print('at end of run_inference iter {}'.format(iteration_num), flush=True)

        return prob

    def run_pycma(self):
        """
            Covariance Matrix Evolution Strategy (CMA-ES)
            Note that the bounds have to be rescaled later, for pycma only.
        """

        if self.retrain_with_counterexamples:
            folder_name = os.path.join(self.package_directory, '../data/experiment4_dist/run_with_retraining')
        else:
            folder_name = os.path.join(self.package_directory, '../data/experiment4_dist/initial_optimization_run')

        self.mug_pipeline.set_folder_name(folder_name)
        self.mug_pipeline.set_optimizer_type(OptimizerType.PYCMA)

        self.mug_initial_poses = []

        for i in range(self.num_mugs):
            self.mug_initial_poses += \
                RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        print(self.mug_initial_poses)

        es = cma.CMAEvolutionStrategy(self.mug_initial_poses, 1.0/3.0,
            {'bounds': [-1.0, 1.0], 'verb_disp': 1, 'popsize': self.num_processes})

        iter_num = 0

        start_time = time.time()
        elapsed_time = 0

        manager = Manager()
        self.all_probabilities = manager.list()
        all_probabilities_lock = manager.Lock()

        self.total_iterations = manager.Value('d', 0)
        self.num_counterexamples = manager.Value('d', 0)

        self.model_number = manager.Value('d', 0)
        model_number_lock = manager.Lock()

        counter_lock = manager.Lock()

        file_q = manager.Queue()

        filename = '{}/results.csv'.format(folder_name)
        watcher = Process(target=self.listener, args=(file_q, filename))
        watcher.start()

        # TODO: share GPU for inference using model.share_memory()

        while not es.stop():
            try:
                ep = EvalParallel3(self.run_inference, number_of_processes=self.num_processes)
                lst = range(iter_num, iter_num + self.num_processes)
                print('lst: {}'.format(lst))
                X = es.ask()
                elapsed_time = time.time() - start_time
                ep(X, lst=lst, args=(self.mug_pipeline, self.all_probabilities,
                    self.total_iterations, self.num_counterexamples,
                    self.model_number, model_number_lock, counter_lock,
                    all_probabilities_lock, file_q),
                    timeout=(self.max_time - elapsed_time))
                print('after ep', flush=True)
            except torch.multiprocessing.context.TimeoutError:
                print('timed out!', flush=True)
                break
            except FoundMaxCounterexamples:
                print('found {} counterexamples!'.format(self.max_counterexamples))
                break
            except Exception as e:
                print("Unhandled exception ", e)
                raise
            except:
                print("Unhandled unnamed exception in pycma")
                raise

            iter_num += self.num_processes
            torch.cuda.empty_cache()
            print('calling ep.terminate()', flush=True)
            ep.terminate()

        elapsed_time = time.time() - start_time
        print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
            elapsed_time/60.0, self.total_iterations.value, elapsed_time/self.total_iterations.value))
        file_q.put('kill')
        print('probabilities:', self.all_probabilities)
        es.result_pretty()

        sys.stdout.flush()

    def run_rbfopt(self):
        """
        Radial Basis Function interpolation.
        """
        folder_name = os.path.join(self.package_directory, '../data/rbfopt/optimization_run')

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

    ## Local optimizers

    @staticmethod
    def run_nelder_mead_helper(process_num, mug_pipeline, num_mugs, all_probabilities,
            total_iterations, num_counterexamples, model_number, model_number_lock,
            counter_lock, all_probabilities_lock, file_q, use_input_initial_poses, mug_initial_poses,
            respawn_when_counterex):

        mug_lower_bound = (-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1)
        mug_upper_bound = (1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2)

        mug_lower_bounds = []
        mug_upper_bounds = []

        for i in range(0, 3):
            mug_lower_bounds += mug_lower_bound
            mug_upper_bounds += mug_upper_bound

        while True:
            try:
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), mug_pipeline.get_folder_name())
                print(folder)
                folder = '{}/optimization_run/{:03d}'.format(folder, process_num)

                if not os.path.exists(folder):
                    print('creating folder')
                    os.mkdir(folder)

                assert(os.path.exists(folder))

                optimize.minimize(mug_pipeline.run_inference, mug_initial_poses, 
                    args=(process_num, all_probabilities, total_iterations, num_counterexamples,
                        model_number, model_number_lock, counter_lock, all_probabilities_lock, file_q,
                        False, respawn_when_counterex),
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

    def run_nelder_mead(self, use_input_initial_poses=False, respawn_when_counterex=False):
        """
        For local search methods, we want to create n parallel processes with random initial poses.
        Each time the process finds a counter example, kill the current process and spawn a
        new process; this allows us to find counter exs that are dissimilar to each other.
        """
        
        mug_initial_poses = []

        if use_input_initial_poses:
            mug_initial_poses = self.mug_initial_poses
        else:
            for i in range(self.num_mugs):
                mug_initial_poses += \
                    RollPitchYaw(np.random.uniform(0.0, 2.0*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                    [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        start_time = time.time()

        manager = Manager()
        self.all_probabilities = manager.list()
        self.total_iterations = manager.Value('d', 0)
        num_counterexamples = manager.Value('d', 0)
        model_number = manager.Value('d', 0)
        model_number_lock = manager.Lock()
        counter_lock = manager.Lock()
        all_probabilities_lock = manager.Lock()

        file_q = manager.Queue()

        folder_name = os.path.join(self.package_directory, "../data/optimization_comparisons/nelder_mead")
        self.mug_pipeline.set_folder_name(folder_name)
        self.mug_pipeline.set_optimizer_type(OptimizerType.NELDER_MEAD)

        pool = Pool(self.num_processes + 1)

        # filename = '{}/results.csv'.format(folder_name)
        # print('filename', filename)
        # watcher = pool.apply_async(self.listener, (file_q, filename))

        filename = '{}/results.csv'.format(folder_name)
        watcher = Process(target=self.listener, args=(file_q, filename))
        watcher.start()

        try:
            result = func_timeout(self.max_time, pool.starmap,
                args=(self.run_nelder_mead_helper,
                zip(range(self.num_processes), repeat(self.mug_pipeline), repeat(self.num_mugs),
                    repeat(self.all_probabilities), repeat(self.total_iterations),
                    repeat(num_counterexamples),
                    repeat(model_number), repeat(model_number_lock),
                    repeat(counter_lock), repeat(all_probabilities_lock),
                    repeat(file_q), repeat(use_input_initial_poses), repeat(mug_initial_poses),
                    repeat(respawn_when_counterex))))
        except FunctionTimedOut:
            elapsed_time = time.time() - start_time
            print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
                elapsed_time/60.0, self.total_iterations.value,
                elapsed_time/self.total_iterations.value))

            file_q.put('kill')
            pool.terminate()
            pool.join()

        end_time = time.time()

        print('probabilities:', self.all_probabilities)

        sys.stdout.flush()

    # SLSQP

    @staticmethod
    def run_slsqp_helper(process_num, mug_pipeline, num_mugs, all_probabilities,
            total_iterations, num_counterexamples, model_number, model_number_lock,
            counter_lock, all_probabilities_lock, file_q, use_input_initial_poses, mug_initial_poses,
            respawn_when_counterex):

        mug_lower_bound = (-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1)
        mug_upper_bound = (1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2)

        mug_lower_bounds = []
        mug_upper_bounds = []

        for i in range(0, 3):
            mug_lower_bounds += mug_lower_bound
            mug_upper_bounds += mug_upper_bound
        mug_bound = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (-0.1, 0.1), (0.1, 0.2)]

        mug_bounds = []
        for i in range(0, 3):
            mug_bounds += mug_bound

        while True:
            try:
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), mug_pipeline.get_folder_name())
                print(folder)
                folder = '{}/optimization_run/{:03d}'.format(folder, process_num)

                if not os.path.exists(folder):
                    print('creating folder')
                    os.mkdir(folder)

                assert(os.path.exists(folder))

                optimize.minimize(mug_pipeline.run_inference, mug_initial_poses, 
                    args=(process_num, all_probabilities, total_iterations, num_counterexamples,
                        model_number, model_number_lock, counter_lock, all_probabilities_lock, file_q,
                        False, respawn_when_counterex),
                    bounds=mug_bounds, method='SLSQP',
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

    def run_slsqp(self, use_input_initial_poses=False, respawn_when_counterex=False):
        """
        For local search methods, we want to create n parallel processes with random initial poses.
        Each time the process finds a counter example, kill the current process and spawn a
        new process; this allows us to find counter exs that are dissimilar to each other.
        """
        
        mug_initial_poses = []

        if use_input_initial_poses:
            mug_initial_poses = self.mug_initial_poses
        else:
            for i in range(self.num_mugs):
                mug_initial_poses += \
                    RollPitchYaw(np.random.uniform(0.0, 2.0*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                    [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

        start_time = time.time()

        manager = Manager()
        self.all_probabilities = manager.list()
        self.total_iterations = manager.Value('d', 0)
        num_counterexamples = manager.Value('d', 0)
        model_number = manager.Value('d', 0)
        model_number_lock = manager.Lock()
        counter_lock = manager.Lock()
        all_probabilities_lock = manager.Lock()

        file_q = manager.Queue()

        folder_name = os.path.join(self.package_directory, "../data/optimization_comparisons/slsqp")
        self.mug_pipeline.set_folder_name(folder_name)
        self.mug_pipeline.set_optimizer_type(OptimizerType.NELDER_MEAD)

        pool = Pool(self.num_processes + 1)

        # filename = '{}/results.csv'.format(folder_name)
        # print('filename', filename)
        # watcher = pool.apply_async(self.listener, (file_q, filename))

        filename = '{}/results.csv'.format(folder_name)
        watcher = Process(target=self.listener, args=(file_q, filename))
        watcher.start()

        try:
            result = func_timeout(self.max_time, pool.starmap,
                args=(self.run_slsqp_helper,
                zip(range(self.num_processes), repeat(self.mug_pipeline), repeat(self.num_mugs),
                    repeat(self.all_probabilities), repeat(self.total_iterations),
                    repeat(num_counterexamples),
                    repeat(model_number), repeat(model_number_lock),
                    repeat(counter_lock), repeat(all_probabilities_lock),
                    repeat(file_q), repeat(use_input_initial_poses), repeat(mug_initial_poses),
                    repeat(respawn_when_counterex))))
        except FunctionTimedOut:
            elapsed_time = time.time() - start_time
            print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
                elapsed_time/60.0, self.total_iterations.value,
                elapsed_time/self.total_iterations.value))

            file_q.put('kill')
            pool.terminate()
            pool.join()

        end_time = time.time()

        print('probabilities:', self.all_probabilities)

        sys.stdout.flush()

    # @staticmethod
    # def run_scipy_fmin_slsqp_helper(mug_pipeline, mug_initial_poses, mug_bounds, all_probabilities, file_q, max_iterations):
    #     exit_mode = optimize.fmin_slsqp(mug_pipeline.run_inference, mug_initial_poses,
    #         bounds=mug_bounds, args=(all_probabilities=all_probabilities, file_q=file_q, increment_iteration_num=True),
    #         full_output=True, iter=max_iterations)
    #     return exit_mode

    # def run_scipy_fmin_slsqp(self, use_input_initial_poses=False):
    #     """
    #         Local optimization (sequential least square programming)
    #     """
    #     folder_name = os.path.join(self.package_directory, '../data/optimization_comparisons/slsqp/optimization_run')

    #     self.mug_pipeline.set_folder_name(folder_name)
    #     self.mug_pipeline.set_optimizer_type(OptimizerType.SLSQP)

    #     mug_initial_poses = []

    #     if use_input_initial_poses:
    #         mug_initial_poses = self.mug_initial_poses
    #     else:
    #         for i in range(self.num_mugs):
    #             mug_initial_poses += \
    #                 RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
    #                 [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

    #     mug_bound = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (-0.1, 0.1), (0.1, 0.2)]

    #     mug_bounds = []
    #     for i in range(0, 3):
    #         mug_bounds += mug_bound

    #     all_probabilities = []

    #     manager = Manager()
    #     file_q = manager.Queue()

    #     filename = '{}/results.csv'.format(folder_name)
    #     watcher = Process(target=self.listener, args=(file_q, filename))
    #     watcher.start()

    #     # exit_mode = optimize.fmin_slsqp(self.mug_pipeline.run_inference, mug_initial_poses,
    #         # bounds=mug_bounds, args=(all_probabilities, file_q),
    #         # full_output=True, iter=self.max_iterations)
    #     exit_mode = self.run_scipy_fmin_slsqp_helper(
    #         self.mug_pipeline, mug_initial_poses, mug_bounds, all_probabilities, file_q, self.max_iterations)

    #     # try:
    #     #     exit_mode = func_timeout(self.max_time, optimize.fmin_slsqp,
    #     #         args=(self.mug_pipeline.run_inference, mug_initial_poses,
    #     #             bounds=mug_bounds, 
    #     #             args=(all_probabilities=all_probabilities, file=file_q),
    #     #             full_output=True, iter=self.max_iterations))
    #     # except FunctionTimedOut:
    #     #     print('fmin_slsqp finished in {} seconds'.format(self.max_time))
    #     # except Exception as e:
    #     #     print('exception {} was thrown'.format(e))

    #     print(exit_mode)
    #     print(all_probabilities)

    ## Metadata and visualization tools

    @staticmethod
    def listener(q, filename):
        """
        Updates csv file with metadata.
        """
        with open(filename, 'w') as f:
            print('opened {}'.format(filename))
            f.write('process_num, iter_num, probability\n')
            while 1:
                m = q.get()
                if m == 'kill':
                    break
                f.write(str(m) + '\n')
                f.flush()

    def plot_graphs(self):
        """
        Plot probabilities
        """

        fig2, ax = plt.subplots()
        print(self.all_probabilities)
        ax.plot(self.all_probabilities)
        ax.set(xlabel='Iteration', ylabel='Probability')
        ax.set_xlim(xmin=0, xmax=len(self.all_probabilities))
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.grid()
        fig2.savefig(os.path.join(self.package_directory, 'probability_plot.png'))
        plt.show()
