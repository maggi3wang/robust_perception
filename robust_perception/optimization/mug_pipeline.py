# Standard libraries imports
import argparse
import codecs
import datetime
from enum import Enum
from itertools import repeat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import shutil
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
from torch.multiprocessing import Process, Pool, Queue, Manager, Lock, Value
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

from ..image_classification.simple_net import SimpleNet
from ..optimization.optimizer import OptimizerType
from ..optimization.model_trainer import MyNet

class FoundCounterexample(Exception):
    pass

class FoundMaxCounterexamples(Exception):
    pass

class ForwardSimulationTimedOut(Exception):
    pass

class NotConstrained(Exception):
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
        # self.label_image_input_port = \
        #     self.DeclareAbstractInputPort("label_image_input_port",
        #         AbstractValue.Make(PydrakeImage[PixelType.kLabel16I](640, 480, 1)))

        self.color_image = None
        # self.label_image = None

    def DoPublish(self, context, event):
        """
        Update color_image and label_image for saving
        """
        self.color_image = self.EvalAbstractInput(context, 0).get_value()
        # self.label_image = self.EvalAbstractInput(context, 1).get_mutable_value()

    def save_image(self, filename):
        """
        Save images to a file
        """
        color_fig = plt.imshow(self.color_image.data)
        plt.axis('off')
        color_fig.axes.get_xaxis().set_visible(False)
        color_fig.axes.get_yaxis().set_visible(False)
        plt.savefig(filename + '_color.png', bbox_inches='tight', pad_inches=0)

        # label_fig = plt.imshow(np.squeeze(self.label_image.data))
        # plt.axis('off')
        # label_fig.axes.get_xaxis().set_visible(False)
        # label_fig.axes.get_yaxis().set_visible(False)
        # plt.savefig(filename + '_label.png', bbox_inches='tight', pad_inches=0)

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

    def __init__(self, num_mugs, max_counterexamples=1000000, generate_counterexample_set=False,
            retrain_with_counterexamples=False, retrain_with_random=False, model_trial_number=0):
        self.num_mugs = num_mugs
        self.initial_poses = None
        self.max_counterexamples = max_counterexamples
        self.optimizer_type = None

        self.final_poses = None
        
        self.folder_name = None
        self.all_poses = []

        self.pose_bundle = None
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

        self.metadata_filename = None

        self.iteration_num = 0
        self.meshcat_visualizer_desired = False

        self.generate_counterexample_set = generate_counterexample_set
        self.retrain_with_counterexamples = retrain_with_counterexamples
        self.retrain_with_random = retrain_with_random

        self.model_trial_number = model_trial_number

        self.is_correct = False
        self.softmax_probabilities = []

        if torch.cuda.is_available():
            cuda.init()
            torch.cuda.set_device(0)
            print(cuda.Device(torch.cuda.current_device()).name(), flush=True)

    def set_optimizer_type(self, optimizer_type):
        self.optimizer_type = optimizer_type

    def get_all_poses(self):
        return self.all_poses

    def set_folder_names(self, folder_name):
        self.folder_name = folder_name

        self.base_training_set_dir = os.path.join(self.folder_name, 'training_set')
        self.initial_training_set_dir = os.path.join(self.base_training_set_dir, 'initial_training_set')
        self.test_set_dir = os.path.join(self.folder_name, 'test_set')
        self.models_dir = os.path.join(self.folder_name, 'models')
        self.counterexample_set_dir = os.path.join(self.folder_name, 'counterexample_set')
        self.trial_training_set_dir = ''

        self.model_prefix = ''
        if self.retrain_with_counterexamples:
            self.model_prefix = 'counterex'
        elif self.retrain_with_random:
            self.model_prefix = 'random'
        self.trial_folder = '{}_{:02d}'.format(self.model_prefix, self.model_trial_number)

        # Create categories in training set directory
        if self.retrain_with_counterexamples or self.retrain_with_random:
            self.trial_training_set_dir = os.path.join(self.base_training_set_dir, self.trial_folder)

            # Create training set directory
            if not os.path.exists(self.trial_training_set_dir):
                os.mkdir(self.trial_training_set_dir)

            for i in range(1, 5 + 1):
                num_mug_trial_training_set_dir = os.path.join(self.trial_training_set_dir, str(i))
                if not os.path.exists(num_mug_trial_training_set_dir):
                    os.mkdir(num_mug_trial_training_set_dir)

        self.training_set_dirs = [self.initial_training_set_dir, self.trial_training_set_dir]

        # Create model directory
        if not os.path.exists(os.path.join(self.models_dir, self.trial_folder)):
            os.mkdir(os.path.join(self.models_dir, self.trial_folder))

        initial_model_dir = os.path.join(self.models_dir, '{}/{}_{:04d}.pth.tar'.format(self.trial_folder, self.trial_folder, 0))
        if not os.path.exists(initial_model_dir):
            shutil.copy(os.path.join(self.folder_name, 'models/initial_00_0000.pth.tar'), initial_model_dir)

        # Create run directory
        self.run_dir = os.path.join(self.folder_name, 'run/{}'.format(self.trial_folder))
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

    def run_meshcat_visualizer(self, builder, scene_graph):
        # Add meshcat visualizer
        blockPrint()
        visualizer = builder.AddSystem(MeshcatVisualizer(
            scene_graph,
            zmq_url="tcp://127.0.0.1:6000",
            draw_period=0.001))
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                visualizer.get_input_port(0))
        enablePrint()

    def write_poses_to_file(self, filename, q0, q0_final):
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

    def create_image(self, iteration_num, process_num=None):
        """
        Create image based on initial poses
        """
        try:
            # print('in create_image - {}'.format(iteration_num), flush=True)

            np.random.seed(46)
            random.seed(46)
            filename = ''

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

            # print('poses: {}'.format(poses), flush=True)
            # print('poses - {}'.format(iteration_num), flush=True)

            if self.meshcat_visualizer_desired:
                self.run_meshcat_visualizer(builder, scene_graph)

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
            # builder.Connect(camera.label_image_output_port(),
            #                 camera_viz.get_input_port(1))

            diagram = builder.Build()
            # print('initialized diagram - {}'.format(iteration_num), flush=True)

            diagram_context = diagram.CreateDefaultContext()
            # print('created diagram_context - {}'.format(iteration_num), flush=True)
            mbp_context = diagram.GetMutableSubsystemContext(
                mbp, diagram_context)
            # print('created mbp_context - {}'.format(iteration_num), flush=True)
            sg_context = diagram.GetMutableSubsystemContext(
                scene_graph, diagram_context)
            # print('created sg_context - {}'.format(iteration_num), flush=True)

            q0 = mbp.GetPositions(mbp_context).copy()
            for k in range(len(poses)):
                offset = k*7
                q0[(offset):(offset+4)] = poses[k][0]
                q0[(offset+4):(offset+7)] = poses[k][1]
            # print('set q0 - {}'.format(iteration_num), flush=True)

            simulator = Simulator(diagram, diagram_context)
            # print('created simulator - {}'.format(iteration_num), flush=True)
            # simulator.set_target_realtime_rate(1.0)
            simulator.set_publish_every_time_step(False)
            # print('set publishing every time step - {}'.format(iteration_num), flush=True)
            simulator.Initialize()
            # print('initialized simulator - {}'.format(iteration_num), flush=True)

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
            # print("Solve info: {}".format(result), flush=True)
            # print("Solved in %f seconds" % (time.time() - start_time))
            # print(result.get_solver_id().name())
            q0_proj = result.GetSolution(q_dec)
            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            # print('q0_initial: {}'.format(q0_initial), flush=True)

            converged = False
            t = 0.1

            start_time = time.time()

            while not converged:
                simulator.AdvanceTo(t)
                t += 0.0001
                
                velocities = mbp.GetVelocities(mbp_context)
                # print(velocities)
                # print('t: {:10.4f}, norm: {:10.4f}, x: {:10.4f}, y: {:10.4f}, z: {:10.4f}'.format(
                #     t, np.linalg.norm(velocities), velocities[0], velocities[1], velocities[2]))

                if np.linalg.norm(velocities) < 0.05:
                    converged = True

                # If haven't timed out in 1 min, want to retry (use another pose)

                if (time.time() - start_time) > 60:
                    print('TIMED OUT IN FORWARD SIMULATION! - {}'.format(iteration_num), flush=True)
                    raise ForwardSimulationTimedOut

            # print('t: {}'.format(t))
            q0_final = mbp.GetPositions(mbp_context).copy()
            # print('q0_final: {}'.format(q0_final), flush=True)

            if self.folder_name is None:
                raise Exception('have not yet set the folder name')

            # TODO fix this horrendous way
            # folder_name = '{}/{}'.format(self.folder_name, 'run_with_retraining')
            # folder_name = '{}/{}/{}'.format(self.run_dir)
            filename = '{}/{}_{:05d}'.format(self.run_dir, n_objects, iteration_num)
            # print(filename)

            # Local optimizer
            if process_num is not None:
                filename = '{}/{:03d}/{}_{:05d}'.format(
                    self.run_dir, process_num, n_objects, iteration_num)

            rgb_and_label_image_visualizer.save_image(filename)

            # Write to a file
            self.write_poses_to_file(filename, q0, q0_final)

            if q0_final[-1] < 0:
                print('POSE IS NOT CONSTRAINED - {}'.format(iteration_num), flush=True)
                raise NotConstrained

            # print('DONE with iteration {}!'.format(self.iteration_num))

            return filename

        except ForwardSimulationTimedOut as e:
            raise ForwardSimulationTimedOut
        except NotConstrained as e:
            raise NotConstrained
        except:
            print("Unhandled unnamed exception, probably sim error")
            raise

    def get_is_correct(self):
        return self.is_correct

    def get_softmax_probabilities(self):
        return self.softmax_probabilities

    def predict_image(self, model, image_path):
        # print('in predict_image image_path: {}'.format(image_path))
        image = None
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print("image isn't in path {}, exception".format(image_path), flush=True)

        image = image.convert('RGB')

        # Define transformations for the image
        transformation = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Preprocess the image
        image_tensor = transformation(image).float()
        # print('image_tensor: ', image_tensor)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        # Turn the input into a Variable
        # input_image = Variable(image_tensor.cuda())       # todo make use_gpu a var
        input_image = Variable(image_tensor)

        # Predict the class of the image
        # with model_number_lock:
        #     # global model
        #     model = q.get()
        #    output = model(input_image)
        output = model(input_image)

        # Add a softmax layer to extract probabilities
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output)

        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        # print('probabilities: {}'.format(probabilities.data.numpy()))

        # print('output data: ', output.data.numpy())
        index = output.data.cpu().numpy().argmax()

        classes = [1, 2, 3, 4, 5]
        is_correct = True

        if classes[index] != self.num_mugs:
            is_correct = False
            print('predicted {} mugs - WRONG, the actual number of mugs is {}!'.format(
                classes[index], self.num_mugs), flush=True)

            f = open(self.metadata_filename, "a")
            f.write('\n----------\n')
            f.write('predicted {} mugs - WRONG, the actual number of mugs is {}!'.format(
                classes[index], self.num_mugs))
            f.close()

        self.is_correct = is_correct
        self.softmax_probabilities = probabilities.data.cpu().numpy()[0]
        return probabilities.data.cpu().numpy()[0], self.is_correct

    def get_folder_name(self):
        return self.folder_name

    def get_iteration_num(self):
        return self.iteration_num

    def retrain(self, num_data_added, model_path):
        """
        todo
        """

        new_net = MyNet(
            self.model_prefix,
            self.model_trial_number,
            num_data_added,
            training_set_dirs=self.training_set_dirs,
            test_set_dir=self.test_set_dir,
            # counterexample_set_dir=self.counterexample_set_dir,
            models_dir=os.path.join(self.models_dir, self.trial_folder))

        new_net.load_and_set_checkpoint(model_path)
        new_net.train(num_epochs=60)

        # model_number_lock.acquire()
        # model_number.value += 1
        # model_number_lock.release()


    # def run_inference(self, poses, iteration_num=None, all_probabilities=[],
    #         total_iterations=Manager().Value('d', 0), num_counterexamples=Manager().Value('d', 0),
    #         model_number=Manager().Value('d', 0), model_number_lock=Lock(), counter_lock=Lock(), all_probabilities_lock=Lock(),
    #         file_q=None, return_is_correct=False, respawn_when_counterex=True):
    def run_inference(self, poses, iteration_num, all_probabilities, all_probabilities_lock,
            total_iterations, num_counterexamples, counter_lock,
            file_q, return_is_correct, respawn_when_counterex):
        """
        Optimizer's entry point function
        It must be a function, not an instancemethod, to work with multiprocessing
        """
        if iteration_num is None:
            iteration_num = self.iteration_num

        # print('iteration_num', iteration_num, flush=True)
        # print('poses', poses)

        if self.optimizer_type is None:
            raise ValueError('Need to set optimizer type before running inference')

        process_num = iteration_num
        if (self.optimizer_type == OptimizerType.NELDER_MEAD or
                self.optimizer_type == OptimizerType.SLSQP):
            iteration_num = self.iteration_num

        if self.optimizer_type == OptimizerType.RANDOM:
            self.iteration_num = iteration_num

        # print('process_num: {}, iteration_num: {}'.format(process_num, iteration_num), flush=True)

        if self.optimizer_type == OptimizerType.PYCMA:
            for i in range(len(poses)):
                if i % 7 == 4 or i % 7 == 5:
                    poses[i] = poses[i] / 10.0
                elif i % 7 == 6:
                    poses[i] = poses[i] / 20.0 + 0.15

        self.initial_poses = poses
        self.all_poses.append(self.initial_poses)
        # print('appended', flush=True)

        # to change for more than one mug
        pose_is_feasible = False
        for i, pose in enumerate(self.initial_poses):
            if pose != 0.0 and i < 4:
                pose_is_feasible = True

        if not pose_is_feasible:
            print('pose is not feasible', flush=True)
            all_probabilities_lock.acquire()
            all_probabilities.append(np.nan)
            all_probabilities_lock.release()
            return 1.01

        # TODO change this, maybe just take in process_num regardless
        try:
            # maybe try more than once if this fails

            if (self.optimizer_type == OptimizerType.NELDER_MEAD or
                    self.optimizer_type == OptimizerType.SLSQP):
                imagefile = self.create_image(iteration_num, process_num)
            else:
                # print('before creating image - {}'.format(iteration_num), flush=True)
                imagefile = self.create_image(iteration_num)
        except Exception as e:
            res = '{:05d}, {:05d}, {}, , , , , , {},'.format(
                    process_num, iteration_num, type(e).__name__, time.time())
            if file_q:
                file_q.put(res)
            else:
                filename = os.path.join(self.folder_name, 'results.csv') 
                f = open(filename, "a+")
                f.write(res)
                f.close()

            print('EXCEPTION {}, returning 1.01 - {}'.format(type(e).__name__, iteration_num), flush=True)
            return 1.01

        imagefile += '_color.png'

        # print('after creating image', flush=True)

        # TODO make this a var
        retrain_batch_size = 50

        counter_lock.acquire()
        model_path = ''
        # TODO clean up this horrendous way
        if self.retrain_with_counterexamples:
            model_path = os.path.join(self.models_dir,
                '{}/{}_{:04d}.pth.tar'.format(
                    self.trial_folder, self.trial_folder,
                    int(num_counterexamples.value / retrain_batch_size) * retrain_batch_size))
        elif self.retrain_with_random:
            model_path = os.path.join(self.models_dir,
                '{}/{}_{:04d}.pth.tar'.format(
                    self.trial_folder, self.trial_folder,
                    int(total_iterations.value / retrain_batch_size) * retrain_batch_size))
        else:
            model_path = os.path.join(self.package_directory,
                '../data/classifier_{:03d}.pth.tar'.format(0))
        counter_lock.release()

        (model, _, _) = MyNet.load_checkpoint(model_path, use_gpu=False)
        model.eval()

        # print('model.eval()', flush=True)

        # Run prediction function and obtain predicted class index
        probabilities, is_correct = self.predict_image(model, imagefile)

        # Return probabilities
        probability = probabilities[self.num_mugs - 1]

        print('iteration: {}, probabilities: {}, probability: {}'.format(
            iteration_num, probabilities, probability), flush=True)
        # print('      {}'.format(self.initial_poses), flush=True)

        ## TODO don't just add to training set, add to counterexs directory so we know which
        # model produced which counterexamples

        # Training set is divided into classes
        training_set_num_dir = os.path.join(self.trial_training_set_dir, '{}'.format(self.num_mugs))

        imagefile_lst = imagefile.split('/')
        imagefile_lst[-1] = self.model_prefix + '_' + imagefile_lst[-1]
        new_imagefile = os.path.join(training_set_num_dir, imagefile_lst[-1])
        
        # Update counter values (total_iterations, num_counterexamples) with lock held
        counter_lock.acquire()
        total_iterations.value += 1

        if not is_correct:
            num_counterexamples.value += 1

        if (self.max_counterexamples is not None and
            num_counterexamples.value >= self.max_counterexamples):
            print('found {} counterexamples'.format(num_counterexamples.value), flush=True)
            raise FoundMaxCounterexamples

        f = open(self.metadata_filename, "a")
        f.write('\n----------\n')
        f.write('iteration: {}, probabilities: {}, probability: {}'.format(
            iteration_num, probabilities, probability))
        f.close()

        if file_q:
            res = '{:05d}, {:05d}, {:1.6f}, {:1.6f}, {:1.6f}, {:1.6f}, {:1.6f}, {:1d}, {},'.format(
                process_num, iteration_num, 
                probabilities[0], probabilities[1], probabilities[2], probabilities[3], probabilities[4],
                is_correct, time.time())
            file_q.put(res)

        if self.retrain_with_random:
            # Move image to training set
            shutil.copy(imagefile, new_imagefile)

            if total_iterations.value % retrain_batch_size == 0:
                print('retraining with random, iterations: {}'.format(total_iterations.value))
                self.retrain(num_data_added=total_iterations.value, model_path=model_path)

        if not is_correct:
            if self.retrain_with_counterexamples:
                print('found {} counterexamples, retraining'.format(num_counterexamples.value, flush=True))

                # Move counterexample to training set
                # Find {train, test, adversarial} set accuracy

                shutil.copy(imagefile, new_imagefile)

                if num_counterexamples.value % retrain_batch_size == 0:
                    print('retraining with counterexamples, num_counterexamples: {}'.format(num_counterexamples.value))
                    self.retrain(num_data_added=num_counterexamples.value, model_path=model_path)
            elif self.generate_counterexample_set:
                # Not retraining, just generating counterexample set
                if not os.path.exists(self.counterexample_set_dir):
                    os.mkdir(self.counterexample_set_dir)

                os.path.join(self.counterexample_set_dir, '{}'.format(self.num_mugs))
                shutil.copy(imagefile, self.counterexample_set_dir)
                print('imagefile: {}, counterexample_set: {}'.format(imagefile, self.counterexample_set_dir),
                    flush=True)
        counter_lock.release()

        all_probabilities_lock.acquire()
        all_probabilities.append(probability)
        all_probabilities_lock.release()

        self.iteration_num += 1

        if self.optimizer_type == OptimizerType.RBFOPT:
            filename = os.path.join(self.folder_name, 'results.csv') 

            f = open(filename, "a+")
            if self.iteration_num == 0:
                f = open(filename, "w+")

            res = '{:05d}, {:05d}, {:1.6f}, {:1.6f}, {:1.6f}, {:1.6f}, {:1.6f}, {:1d}, {},\n'.format(
                process_num, iteration_num, 
                probabilities[0], probabilities[1], probabilities[2], probabilities[3], probabilities[4],
                is_correct, time.time())
            f.write(res)
            f.close()

        print('probability: {}'.format(probability), flush=True)
        sys.stdout.flush()

        if not is_correct and respawn_when_counterex:
            print('raising FoundCounterexample exception')
            raise FoundCounterexample

        if return_is_correct:
            return probability, is_correct

        return probability
