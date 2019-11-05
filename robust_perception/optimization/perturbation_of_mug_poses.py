# TODO move this!!!
from functools import partial
import numpy as np
from multiprocessing import Pool as ProcessingPool
import warnings

class EvalParallel3(object):
    def __init__(self, fitness_function=None, number_of_processes=None):
        self.fitness_function = fitness_function
        self.processes = number_of_processes
        self.pool = ProcessingPool(self.processes)

    def __call__(self, solutions, fitness_function=None, lst=None, args=(), timeout=None):
        """evaluate a list/sequence of solution-"vectors", return a list
        of corresponding f-values.
        Raises `multiprocessing.TimeoutError` if `timeout` is given and
        exceeded.
        """
        fitness_function = fitness_function or self.fitness_function
        if fitness_function is None:
            raise ValueError("`fitness_function` was never given, must be"
                             " passed in `__init__` or `__call__`")
        warning_str = ("WARNING: `fitness_function` must be a function,"
                       " not an instancemethod, in order to work with"
                       " `multiprocessing`")
        if isinstance(fitness_function, type(self.__init__)):
            warnings.warn(warning_str)
        jobs = [self.pool.apply_async(fitness_function, (x,iter_num) + args)
                for x, iter_num in zip(solutions, lst)]
        try:
            return [job.get(timeout) for job in jobs]
        except:
            warnings.warn(warning_str)
            raise

    def terminate(self):
        """free allocated processing pool"""
        # self.pool.close()  # would wait for job termination
        self.pool.terminate()  # terminate jobs regardless
        self.pool.join()  # end spawning

    def __enter__(self):
        # we could assign self.pool here, but then `EvalParallel2` would
        # *only* work when using the `with` statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __del__(self):
        """though generally not recommended `__del__` should be OK here"""
        self.terminate()


# Standard libraries imports
import argparse
import codecs
import datetime
from itertools import repeat
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool, Queue, Manager, Lock, Value
import multiprocessing
import numpy as np
import os
import random
import time
import yaml
import sys

# pydrake imports
from func_timeout import func_timeout, FunctionTimedOut

import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere
)
from pydrake.math import (RollPitchYaw, RigidTransform)
from pydrake.multibody.tree import (
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)

from pydrake.forwarddiff import gradient
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mathematicalprogram import (SolverOptions)
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.solvers.nlopt import (NloptSolver)
from pydrake.solvers.snopt import (SnoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem, PortDataType
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import RgbdSensor, Image as PydrakeImage, PixelType, PixelFormat
from pydrake.geometry.render import DepthCameraProperties, MakeRenderEngineVtk, RenderEngineVtkParams

import torch
import torch.nn as nn
import pycuda.driver as cuda
from torchvision.transforms import transforms
from torch.autograd import Variable
import requests
import shutil
from io import open, BytesIO
from PIL import Image, ImageFile
import json

# Optimization library imports
import nevergrad as ng
import rbfopt
from scipy import optimize
import cma
from cma.fitness_transformations import EvalParallel2

# Local imports
from ..image_classification.simple_net import SimpleNet


class FoundCounterexample(Exception):
    pass

class RgbAndLabelImageVisualizer(LeafSystem):
    def __init__(self, draw_timestep=0.033333):
        LeafSystem.__init__(self)
        self.set_name('image viz')
        self.timestep = draw_timestep
        self.DeclarePeriodicPublish(draw_timestep, 0.0)
        
        self.rgb_image_input_port = \
            self.DeclareAbstractInputPort("rgb_image_input_port",
                AbstractValue.Make(PydrakeImage[PixelType.kRgba8U](640, 480, 3)))
        self.label_image_input_port = \
            self.DeclareAbstractInputPort("label_image_input_port",
                AbstractValue.Make(PydrakeImage[PixelType.kLabel16I](640, 480, 1)))

        self.color_image = None
        self.label_image = None

    def DoPublish(self, context, event):
        """
        Update color_image and label_image for saving
        """
        self.color_image = self.EvalAbstractInput(context, 0).get_value()
        self.label_image = self.EvalAbstractInput(context, 1).get_mutable_value()

    def save_image(self, filename):
        """
        Save images to a file
        """
        color_fig = plt.imshow(self.color_image.data)
        plt.axis('off')
        color_fig.axes.get_xaxis().set_visible(False)
        color_fig.axes.get_yaxis().set_visible(False)
        plt.savefig(filename + '_color.png', bbox_inches='tight', pad_inches=0)

        label_fig = plt.imshow(np.squeeze(self.label_image.data))
        plt.axis('off')
        label_fig.axes.get_xaxis().set_visible(False)
        label_fig.axes.get_yaxis().set_visible(False)
        plt.savefig(filename + '_label.png', bbox_inches='tight', pad_inches=0)

def blockPrint():
    """
    Disable print
    """
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """
    Enable print
    """
    sys.stdout = sys.__stdout__


class MugPipeline():
    def __init__(self, num_mugs, initial_poses):
        self.num_mugs = num_mugs
        self.initial_poses = initial_poses

        self.final_poses = None
        
        self.folder_name = None
        self.all_poses = []
        # self.all_probabilities = []
        self.pose_bundle = None
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

        self.metadata_filename = None

        self.iteration_num = 0

        if torch.cuda.is_available():
            cuda.init()
            torch.cuda.set_device(0)
            print(cuda.Device(torch.cuda.current_device()).name())

    def get_all_poses(self):
        return self.all_poses

    # def get_all_probabilities(self):
    #     global all_probabilities
    #     return all_probabilities

    def set_folder_name(self, folder_name):
        self.folder_name = folder_name

    def create_image(self, iteration_num, process_num=None):
        """
        Create image based on initial poses
        """
        np.random.seed(46)
        random.seed(46)
        filename = ''

        try:
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.0001))
            renderer_params = RenderEngineVtkParams()
            scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(renderer_params))

            # Add ground
            world_body = mbp.world_body()
            ground_shape = Box(2., 2., 2.)
            ground_body = mbp.AddRigidBody("ground", SpatialInertia(
                mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
            
            mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                           RigidTransform(Isometry3(rotation=np.eye(3), translation=[0, 0, -1])))
            mbp.RegisterVisualGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
                np.array([0.5, 0.5, 0.5, 1.]))
            mbp.RegisterCollisionGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
                CoulombFriction(0.9, 0.8))

            parser = Parser(mbp, scene_graph)

            os.path.join(self.package_directory, '../image_classification/')

            candidate_model_files = [
                os.path.join(self.package_directory, '../dataset_generation/mug_clean/mug.urdf')
            ]

            n_objects = self.num_mugs
            poses = []  # [quat, pos]
            classes = []
            for k in range(n_objects):
                model_name = "model_%d" % k
                model_ind = np.random.randint(0, len(candidate_model_files))
                class_path = candidate_model_files[model_ind]
                classes.append(class_path)
                parser.AddModelFromFile(class_path, model_name=model_name)
                poses.append(
                    [np.array(
                        [self.initial_poses[7*k + 0], self.initial_poses[7*k + 1],
                        self.initial_poses[7*k + 2], self.initial_poses[7*k + 3]]),
                        self.initial_poses[7*k + 4], self.initial_poses[7*k + 5],
                        self.initial_poses[7*k + 6]])

            mbp.Finalize()

            # print(poses)

            # Add meshcat visualizer
            # blockPrint()
            # visualizer = builder.AddSystem(MeshcatVisualizer(
            #     scene_graph,
            #     zmq_url="tcp://127.0.0.1:6000",
            #     draw_period=0.001))
            # builder.Connect(scene_graph.get_pose_bundle_output_port(),
            #         visualizer.get_input_port(0))
            # enablePrint()

            # Add camera
            depth_camera_properties = DepthCameraProperties(
                width=1000, height=1000, fov_y=np.pi/2, renderer_name="renderer", z_near=0.1, z_far=2.0)
            parent_frame_id = scene_graph.world_frame_id()
            camera_tf = RigidTransform(p=[0.0, 0.0, 0.95], rpy=RollPitchYaw([0, np.pi, 0]))
            camera = builder.AddSystem(
                RgbdSensor(parent_frame_id, camera_tf, depth_camera_properties, show_window=False))
            camera.DeclarePeriodicPublish(0.1, 0.)
            builder.Connect(scene_graph.get_query_output_port(),
                            camera.query_object_input_port())

            rgb_and_label_image_visualizer = RgbAndLabelImageVisualizer(draw_timestep=0.1)
            camera_viz = builder.AddSystem(rgb_and_label_image_visualizer)
            builder.Connect(camera.color_image_output_port(),
                            camera_viz.get_input_port(0))
            builder.Connect(camera.label_image_output_port(),
                            camera_viz.get_input_port(1))

            diagram = builder.Build()

            diagram_context = diagram.CreateDefaultContext()
            mbp_context = diagram.GetMutableSubsystemContext(
                mbp, diagram_context)
            sg_context = diagram.GetMutableSubsystemContext(
                scene_graph, diagram_context)

            q0 = mbp.GetPositions(mbp_context).copy()
            for k in range(len(poses)):
                offset = k*7
                q0[(offset):(offset+4)] = poses[k][0]
                q0[(offset+4):(offset+7)] = poses[k][1]

            simulator = Simulator(diagram, diagram_context)
            # simulator.set_target_realtime_rate(1.0)
            simulator.set_publish_every_time_step(False)
            simulator.Initialize()
            
            ik = InverseKinematics(mbp, mbp_context)
            q_dec = ik.q()
            prog = ik.get_mutable_prog()

            def squaredNorm(x):
                return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])

            for k in range(len(poses)):
                # Quaternion norm
                prog.AddConstraint(
                    squaredNorm, [1], [1], q_dec[(k*7):(k*7+4)])
                # Trivial quaternion bounds
                prog.AddBoundingBoxConstraint(
                    -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
                # Conservative bounds on on XYZ
                prog.AddBoundingBoxConstraint(
                    np.array([-2., -2., -2.]), np.array([2., 2., 2.]),
                    q_dec[(k*7+4):(k*7+7)])

            def vis_callback(x):
                mbp.SetPositions(mbp_context, x)
                global pose_bundle
                pose_bundle = scene_graph.get_pose_bundle_output_port().Eval(sg_context)

            prog.AddVisualizationCallback(vis_callback, q_dec)
            prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

            ik.AddMinimumDistanceConstraint(0.001, threshold_distance=1.0)
            
            prog.SetInitialGuess(q_dec, q0)
            start_time = time.time()
            solver = SnoptSolver()
            sid = solver.solver_type()
            # prog.SetSolverOption(sid, "Print file", "test.snopt")
            prog.SetSolverOption(sid, "Major feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Major optimality tolerance", 1e-2)
            prog.SetSolverOption(sid, "Minor feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Scale option", 0)

            # print("Solver opts: ", prog.GetSolverOptions(solver.solver_type()))
            # print(type(prog))
            result = mp.Solve(prog)
            # print("Solve info: ", result)
            # print("Solved in %f seconds" % (time.time() - start_time))
            # print(result.get_solver_id().name())
            q0_proj = result.GetSolution(q_dec)
            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            # print('q0_initial: ', q0_initial)

            converged = False
            t = 0.1

            while not converged:
                simulator.AdvanceTo(t)
                t += 0.0001
                
                velocities = mbp.GetVelocities(mbp_context)
                # print(velocities)
                # print('t: {:10.4f}, norm: {:10.4f}, x: {:10.4f}, y: {:10.4f}, z: {:10.4f}'.format(
                #     t, np.linalg.norm(velocities), velocities[0], velocities[1], velocities[2]))

                if np.linalg.norm(velocities) < 0.05:
                    converged = True

            q0_final = mbp.GetPositions(mbp_context).copy()
            # print('q0_final: ', q0_final)

            if self.folder_name is None:
                raise Exception('have not yet set the folder name')

            filename = 'robust_perception/optimization/{}/{:05d}_{}'.format(
                self.folder_name, iteration_num, n_objects)

            if process_num is not None:
                filename = 'robust_perception/optimization/{}/{:03d}/{:05d}_{}'.format(
                    self.folder_name, process_num, iteration_num, n_objects)                

            # time.sleep(0.5)
            rgb_and_label_image_visualizer.save_image(filename)

            self.metadata_filename = filename + '_metadata.txt'
            f = open(self.metadata_filename, "w+")

            self.after_solver_poses = q0.flatten()
            self.final_poses = q0_final.flatten()

            def divide_chunks(l, n):     
                # looping till length l 
                for i in range(0, len(l), n):  
                    yield l[i:i + n] 

            n = 7

            self.initial_poses = list(divide_chunks(self.initial_poses, n))
            self.after_solver_poses = list(divide_chunks(self.after_solver_poses, n))
            self.final_poses = list(divide_chunks(self.final_poses, n))

            for pose in self.initial_poses:
                for count, item in enumerate(pose): 
                    f.write("%8.8f " % item)
                f.write("\n")

            f.write('\n----------\n')

            for pose in self.after_solver_poses:
                for count, item in enumerate(pose): 
                    f.write("%8.8f " % item)
                f.write("\n")

            f.write('\n----------\n')

            for pose in self.final_poses:
                for count, item in enumerate(pose): 
                    f.write("%8.8f " % item)
                f.write("\n")

            f.close()

            # print('DONE with iteration {}!'.format(self.iteration_num))
            # time.sleep(5.0)

        except Exception as e:
            print("Unhandled exception ", e)
            raise

        except:
            print("Unhandled unnamed exception, probably sim error")
            raise

        return filename

    def predict_image(self, model, image_path):
        # print('in predict_image image_path: {}'.format(image_path))
        image = None
        try:
            image = Image.open(image_path + '_color.png')
        except FileNotFoundError:
            print("image isn't in path {}, exception".format(image_path))

        image = image.convert('RGB')

        # max_edge_length = 369   # TODO find this programatically

        # Define transformations for the image
        transformation = transforms.Compose([
            # transforms.CenterCrop((max_edge_length)),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess the image
        image_tensor = transformation(image).float()
        # print('image_tensor: ', image_tensor)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        image_tensor.cuda()

        # Turn the input into a Variable
        input = Variable(image_tensor)

        # Predict the class of the image
        output = model(input)

        # Add a softmax layer to extract probabilities
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output)

        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        # print('probabilities: {}'.format(probabilities.data.numpy()))

        # print('output data: ', output.data.numpy())
        index = output.data.numpy().argmax()

        classes = [1, 2, 3, 4, 5]

        word = 'are'
        s = 's'
        if classes[index] == 1:
            word = 'is'
            s = ''

        # print('there {} {} mug{}'.format(word, classes[index], s))

        is_correct = True

        if classes[index] != self.num_mugs:
            is_correct = False
            print('predicted {} mugs - WRONG, the actual number of mugs is {}!'.format(
                classes[index], self.num_mugs))

            f = open(self.metadata_filename, "a")
            f.write('\n----------\n')
            f.write('predicted {} mugs - WRONG, the actual number of mugs is {}!'.format(
                classes[index], self.num_mugs))
            f.close()

        return probabilities.data.numpy()[0], is_correct

    def run_inference(self, poses, iteration_num, all_probabilities,
            total_iterations=None, total_iterations_lock=None):
        """
        Optimizer's entry point function
        It must be a function, not an instancemethod, to work with multiprocessing
        """

        # print('iteration_num', iteration_num)
        # print('poses', poses)

        # change this horrendous way
        process_num = iteration_num
        if 'nelder_mead' in self.folder_name:
            iteration_num = self.iteration_num

        print('process_num: {}, iteration_num: {}'.format(process_num, iteration_num))

        path = os.path.join(self.package_directory,
            '../image_classification/mug_numeration_classifier.model')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = SimpleNet(num_classes=5)
        model.load_state_dict(checkpoint)
        model.eval()

        if "pycma" in self.folder_name:
            for i in range(len(poses)):
                if i % 7 == 4 or i % 7 == 5:
                    poses[i] = poses[i] / 10.0
                elif i % 7 == 6:
                    poses[i] = poses[i] / 20.0 + 0.15

        self.initial_poses = poses
        self.all_poses.append(self.initial_poses)

        # to change for more than one mug
        pose_is_feasible = False
        for i, pose in enumerate(self.initial_poses):
            if pose != 0.0 and i < 4:
                pose_is_feasible = True

        if not pose_is_feasible:
            all_probabilities.append(np.nan)
            return 1.01

        # TODO change this, maybe just take in process_num regardless
        if 'nelder_mead' in self.folder_name:
            imagefile = self.create_image(iteration_num, process_num)
        else:
            imagefile = self.create_image(iteration_num)

        # Run prediction function and obtain predicted class index
        probabilities, is_correct = self.predict_image(model, imagefile)

        # Return probabilities
        probability = probabilities[self.num_mugs - 1]

        print('iteration: {}, probabilities: {}, probability: {}'.format(
            iteration_num, probabilities, probability))
        print('      {}'.format(self.initial_poses))

        f = open(self.metadata_filename, "a")
        f.write('\n----------\n')
        f.write('iteration: {}, probabilities: {}, probability: {}'.format(
            iteration_num, probabilities, probability))
        f.close()

        # global all_probabilities
        all_probabilities.append(probability)

        with total_iterations_lock:
            total_iterations.value += 1

        if 'nelder_mead' in self.folder_name:
            self.iteration_num += 1

            if not is_correct:
                print('raising FoundCounterexample exception')
                raise FoundCounterexample

        sys.stdout.flush()

        return probability


class Optimizer():
    highest_process_num = 20

    def __init__(self, num_mugs, mug_initial_pose, mug_lower_bound, mug_upper_bound,
            max_evaluations, max_time):
        self.mug_initial_poses = []
        for _ in range(num_mugs):
            self.mug_initial_poses += mug_initial_pose

        self.mug_lower_bounds = []
        for _ in range(num_mugs):
            self.mug_lower_bounds += mug_lower_bound

        self.mug_upper_bounds = []
        for _ in range(num_mugs):
            self.mug_upper_bounds += mug_upper_bound

        self.mug_pipeline = MugPipeline(num_mugs=num_mugs, initial_poses=self.mug_initial_poses)
        self.num_mugs = num_mugs
        self.max_evaluations = max_evaluations
        self.max_time = max_time         # [s]

        self.num_vars = 7 * self.num_mugs
        self.num_processes = Optimizer.highest_process_num

        self.all_probabilities = None

        # TODO make this global
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 32) & (2**32 - 1))
        random.seed(os.urandom(4))

    @staticmethod
    def run_inference(poses, iteration_num, mug_pipeline, all_probabilities):
        """
        Wrapper for optimizer's entry point function
        """
        prob = mug_pipeline.run_inference(poses, iteration_num, all_probabilities)
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
        settings = rbfopt.RbfoptSettings(max_evaluations=self.max_evaluations)
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

        self.mug_pipeline.set_folder_name("data_pycma")

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

        while not es.stop():
            try:
                ep = EvalParallel3(self.run_inference, number_of_processes=self.num_processes)
                lst = range(iter_num, iter_num + self.num_processes)
                X = es.ask()
                elapsed_time = time.time() - start_time
                ep(X, lst=lst, args=(self.mug_pipeline, self.all_probabilities),
                    timeout=(self.max_time - elapsed_time))
            except multiprocessing.context.TimeoutError:
                print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
                    elapsed_time/60.0, iter_num, elapsed_time/iter_num))
                break

            iter_num += self.num_processes
            torch.cuda.empty_cache()
            ep.terminate()

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
            bounds=mug_bounds, full_output=True, iter=self.max_evaluations)

        print(exit_mode)

    @staticmethod
    def run_nelder_mead_process(process_num, mug_pipeline, num_mugs, all_probabilities,
            total_iterations, total_iterations_lock):
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
                    args=(process_num, all_probabilities, total_iterations, total_iterations_lock),
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
                process_num = Optimizer.highest_process_num

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

        self.mug_pipeline.set_folder_name("data_scipy_nelder_mead")
        pool = Pool(self.num_processes)

        try:
            result = func_timeout(self.max_sec, pool.starmap,
                args=(self.run_nelder_mead_process,
                zip(range(self.num_processes), repeat(self.mug_pipeline), repeat(self.num_mugs),
                    repeat(self.all_probabilities), repeat(self.total_iterations), repeat(total_iterations_lock))))

        except FunctionTimedOut:
            elapsed_time = time.time() - start_time
            print('ran for {} minutes! total number of iterations is {}, with {} sec/image'.format(
                elapsed_time/60.0, self.total_iterations.value, elapsed_time/self.total_iterations.value))

            pool.terminate()
            pool.join()

        end_time = time.time()

        print('probabilities:', self.all_probabilities)

        sys.stdout.flush()

def main():
    mug_initial_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]
    max_sec = 60.0 * 60.0 * 5.0
    optimizer = Optimizer(num_mugs=3, mug_initial_pose=mug_initial_pose,
        mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound, max_evaluations=50000,
        max_time=max_sec)

    multiprocessing.set_start_method('spawn')

    ## Global optimizers

    # optimizer.run_rbfopt()
    # optimizer.plot_graphs()

    # Run all the optimizers
    # optimizer.plot_graphs(optimizer.run_nevergrad())

    # optimizer.run_pycma()
    # optimizer.plot_graphs()

    ## Local optimizers

    # optimizer.run_scipy_fmin_slsqp()
    # optimizer.plot_graphs()

    optimizer.run_scipy_nelder_mead()
    optimizer.plot_graphs()

if __name__ == "__main__":
    main()
