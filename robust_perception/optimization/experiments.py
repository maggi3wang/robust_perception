import argparse
import codecs
import datetime
from enum import Enum
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

from torch.multiprocessing import Process, Pool, Queue, Manager, Lock, Value
import torch.multiprocessing

# Local imports
from ..optimization.eval_parallel import EvalParallel3
from ..optimization.mug_pipeline import MugPipeline, FoundCounterexample, FoundMaxCounterexamples
from ..optimization.model_trainer import MyNet
from ..optimization.optimizer import OptimizerType

# def run_inference(self, poses, iteration_num, all_probabilities=None,
#     total_iterations=None, num_counterexamples=None,
#     model_number=0, model_number_lock=None, counter_lock=None, all_probabilities_lock=None,
#     file_q=None):

class Experiment():
    def __init__(self, num_mugs):
        self.num_mugs = num_mugs

    #@staticmethod
    #def run_inference(poses, iter_num):


    def run_experiment(self):
        k = 1000        # this is num_samples - 1

        pose_k = [0.59511603, 0.19362723, -0.65069286, 0.43005140, -0.07855161, 0.04501751, 0.16365936, 
            0.04717458, 0.55405918, -0.25733543, -0.79029834, -0.07334251, -0.06119629, 0.14028863, 
            0.51274931, -0.27527083, 0.50746309, 0.63544891, 0.01261620, -0.05093206, 0.10193130, ] 

        # Use parallelized processes to find counterex pose
        # pose_k = []     # will change pose_k to be counterexample pose
        found_counterex = False

        mug_pipeline = MugPipeline(num_mugs=self.num_mugs)

        # folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #    '../data/experiment1/find_counterexample')
        # mug_pipeline.set_folder_name(folder_name)
        mug_pipeline.set_optimizer_type(OptimizerType.NONE)

        num_processes = 20
        pool = Pool(processes=num_processes)
        manager = Manager()
        model_number = manager.Value('d', 0)
        # iter_num = 0

        # while not found_counterex:
        #     sample_pose_ks = []
        #     for i in range(num_processes):
        #         pose = []
        #         for x in range(self.num_mugs):
        #             pose.extend(RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
        #                 [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)])
        #         sample_pose_ks.append(pose)

        #     lst = range(iter_num, iter_num + num_processes)
        #     jobs = [pool.apply_async(self.run_inference, (pose, i, mug_pipeline)) for pose, i in zip(sample_pose_ks, lst)]

        #     for job in jobs:
        #         try:
        #             job.get()
        #         except BaseException:
        #             counterex_iter_num = mug_pipeline.get_iteration_num()
        #             print('counterex_iter_num', counterex_iter_num)
        #             pose_k = sample_pose_ks[counterex_iter_num % num_processes]
        #             break

        #     iter_num += num_processes
        #     # pose_k = []
        #     # for x in range(self.num_mugs):
        #     #     pose_k.extend(RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
        #     #         [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)])

        #     # pose_k_probability = mug_pipeline.run_inference(pose_k, k)
        #     # found_counterex = not(mug_pipeline.get_is_correct())

        # print('iter_num', iter_num)

        # Sample initial posesp
        pose_0 = []
        for x in range(self.num_mugs):
            pose_0.extend(RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
                [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)])

        pose_0 = np.array(pose_0)
        pose_k = np.array(pose_k)
    
        print(pose_0)
        print(pose_k)

        #a_list = []
        # probability_list = []
        pose_diff = pose_k - pose_0

        folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../data/experiment1/initial_optimization_run')
        mug_pipeline.set_folder_name(folder_name)

        i = 0

        #while i < k + 1:
        #    lst = np.array(range(i, i + num_processes)) / float(k)
        #    a_list.extend(a)
        a_list = np.array(range(0, k+1)) / float(k)
        pose_list = []
        for a in a_list:
            pose_list.append(list(pose_0 + a * pose_diff))

        print('pose_list', pose_list)
        print('a_list', a_list)
        probability_list = pool.starmap(mug_pipeline.run_inference, zip(pose_list, range(0, len(a_list))))


        #for i in range(0, k + 1):
        #    a = float(i) / k        # a is btwn 0 and 1 (0 for x0, 1 for xk)
        #    a_list.append(a)
        #    pose = pose_0 + a * pose_diff
        #    probability_list.append(mug_pipeline.run_inference(pose, i))

        plt.close()
        fig2, ax = plt.subplots()
        print(a_list, probability_list)
        ax.scatter(range(0, len(a_list)), probability_list, markersize=10)
        ax.set(xlabel='Iteration', ylabel='Probability')
        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0.0, ymax=1.0)
        ax.grid()
        fig2.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../data/experiment1/initial_optimization_run/probability_plot.png'))
        plt.show()
