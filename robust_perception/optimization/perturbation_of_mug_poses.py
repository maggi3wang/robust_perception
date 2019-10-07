# Standard libraries imports
import argparse
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

# pydrake imports
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

# Local imports
from ..image_classification.simple_net import SimpleNet


class RgbAndLabelImageVisualizer(LeafSystem):
    def __init__(self, draw_timestep=0.00001):
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
        self.all_probabilities = []
        self.iteration_num = 0
        self.pose_bundle = None
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

    def get_all_poses(self):
        return self.all_poses

    def get_all_probabilities(self):
        return self.all_probabilities

    def set_folder_name(self, folder_name):
        self.folder_name = folder_name

    def create_image(self):
        """
        Create image based on initial poses
        """
        np.random.seed(46)
        random.seed(46)
        filename = ''

        try:
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.001))
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

            # Add meshcat visualizer
            # blockPrint()
            # visualizer = builder.AddSystem(MeshcatVisualizer(
            #     scene_graph,
            #     zmq_url="tcp://127.0.0.1:6000",
            #     draw_period=0.0001))
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

            rgb_and_label_image_visualizer = RgbAndLabelImageVisualizer(draw_timestep=0.0001)
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
            # simulator.AdvanceTo(10.0)
            converged = False
            t = 0.0

            while not converged:
                t += 0.001
                simulator.AdvanceTo(t)
                
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

            filename = 'robust_perception/optimization/{}/{:04d}_{}'.format(
                self.folder_name, self.iteration_num, n_objects)
            # time.sleep(0.5)
            rgb_and_label_image_visualizer.save_image(filename)

            metadata_filename = filename + '_metadata.txt'
            f = open(metadata_filename, "w+")

            self.initial_poses = q0.flatten()
            self.final_poses = q0_final.flatten()

            def divide_chunks(l, n):     
                # looping till length l 
                for i in range(0, len(l), n):  
                    yield l[i:i + n] 

            n = 7
            self.initial_poses = list(divide_chunks(self.initial_poses, n))
            self.final_poses = list(divide_chunks(self.final_poses, n))

            for pose in self.initial_poses:
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

        max_edge_length = 369   # TODO find this programatically

        # Define transformations for the image
        transformation = transforms.Compose([
            transforms.CenterCrop((max_edge_length)),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess the image
        image_tensor = transformation(image).float()
        # print('image_tensor: ', image_tensor)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
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

        if classes[index] != self.num_mugs:
            print('predicted {} mugs - WRONG, the actual number of mugs is {}!'.format(
                classes[index], self.num_mugs))
            # TODO in this case, don't just use the highest -- it needs to be the right mug
        # else:
        #     print('this is correct')

        return probabilities.data.numpy()[0]

    def run_inference(self, poses):
        """
        Optimizer's entry point function
        """

        path = os.path.join(self.package_directory,
            '../image_classification/mug_numeration_classifier.model')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = SimpleNet(num_classes=5)
        model.load_state_dict(checkpoint)
        model.eval()

        # print('POSES:', poses)
        self.initial_poses = poses
        self.all_poses.append(self.initial_poses)

        # to change for more than one mug
        pose_is_feasible = False
        for i, pose in enumerate(self.initial_poses):
            if pose != 0.0 and i < 4:
                pose_is_feasible = True

        if not pose_is_feasible:
            self.all_probabilities.append(np.nan)
            return 1.01

        imagefile = self.create_image()

        # Run prediction function and obtain predicted class index
        probabilities = self.predict_image(model, imagefile)

        # Return probabilities
        probability = probabilities[self.num_mugs - 1]
        print('iteration: {}, probabilities: {}, probability: {}'.format(
            self.iteration_num, probabilities, probability))
        print('      {}'.format(self.initial_poses))
        self.iteration_num += 1
        self.all_probabilities.append(probability)
        return probability


class Optimizer():
    def __init__(self, num_mugs, mug_initial_pose, mug_lower_bound, mug_upper_bound,
            max_evaluations):
        """
        
        Parameters:
        - self
        - optimizer_type

        """
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

        self.num_vars = 7 * self.num_mugs

        # TODO make this global
        self.package_directory = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def run_inference(poses):
        """
        Wrapper for optimizer's entry point function
        """
        return self.mug_pipeline.run_inference(poses)

    # def run_optimizer():
    #     if optimizer_type == 'nevergrad':
    #         plot_graphs(run_nevergrad())
    #     else:
    #         raise Exception('optimizer type not defined')

    def plot_graphs(self):
        fig2, ax = plt.subplots()
        all_probabilities = self.mug_pipeline.get_all_probabilities()
        ax.plot(all_probabilities)
        ax.set(xlabel='Iteration', ylabel='Probability')
        ax.set_xlim(xmin=0, xmax=len(all_probabilities))
        ax.set_ylim(ymin=min(all_probabilities), ymax=1.0)
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

    # def run_pycma(self):
    #     """

    #     """
    #     mug_pipeline.set_folder_name("data_pycma")

    #     pass

    # def run_scipy_fmin_slsqp(self):
    #     """

    #     """
    #     mug_pipeline.set_folder_name("data_fmin_slsqp")

    #     # mug_initials = [0.80318661, 0.23925861, -0.17375544, -0.51716113, -0.05577754, -0.00002082, 0.18672807, 
    #     #     -0.82975143, -0.12051900, -0.03752475, -0.54367236, -0.05793993, -0.07058323, 0.13473190, 
    #     #     0.60546342, -0.38073887, 0.65221595, 0.25113008, 0.04118729, 0.05237785, 0.16974618]

    #     ##### FIGURE OUT WHY setting this gives a configuration error
    #     # this should work?
    #     mug_initial = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #     # mug_lower_bound = (-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1)
    #     # mug_upper_bound = (1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2)
    #     # mug_bounds = ((-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1), (1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2))
    #     mug_bound = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (-0.1, 0.1), (0.1, 0.2)]

    #     mug_initials = []
    #     mug_bounds = []
    #     # mug_lower_bounds = []
    #     # mug_upper_bounds = []
    #     for i in range(0, 3):
    #         mug_initials += mug_initial
    #         mug_bounds += mug_bound
    #         # mug_lower_bounds += mug_lower_bound
    #         # mug_upper_bounds += mug_upper_bound

    #     # print(mug_initials)
    #     # print(mug_bounds)
    #     # print(mug_lower_bounds)
    #     # print(mug_upper_bounds)

    #     # optimize.minimize(run_inference, mug_initials, bounds=(mug_lower_bounds, mug_upper_bounds),
    #     #     method='Powell', options={'maxiter':1000, 'disp': True}) # Powell, "Nelder-Mead") method='BFGS')

    #     exit_mode = optimize.fmin_slsqp(run_inference, mug_initials, bounds=mug_bounds, full_output=True, iter=500)
    #     # maybe change epsilon?

    #     print(exit_mode)

    #     # objval, x, itercount, evalcount, fast_evalcount = alg.optimize()
    #     # state_path = os.path.join(package_directory, 'state.dat')
    #     # print(state_path)
    #     # alg.save_to_file(state_path)

    #     global all_poses
    #     global all_probabilities

    #     print('all poses:', all_poses)
    #     print('all probabilities', all_probabilities)

    #     fig2, ax = plt.subplots()
    #     ax.plot(all_probabilities)
    #     ax.set(xlabel='Iteration', ylabel='Probability')
    #     ax.set_xlim(xmin=0, xmax=31)
    #     ax.set_ylim(ymin=0.95, ymax=1.0)
    #     ax.grid()
    #     fig2.savefig(os.path.join(package_directory, 'probability_plot.png'))
    #     plt.show()

def main():
    mug_initial_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]
    optimizer = Optimizer(num_mugs=3, mug_initial_pose=mug_initial_pose,
        mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound, max_evaluations=300)

    # Run all the optimizers
    # optimizer.plot_graphs(optimizer.run_nevergrad())

    optimizer.run_rbfopt()
    optimizer.plot_graphs()
    # optimizer.run_pycma()
    # optimizer.run_scipy_fmin_slsqp()

if __name__ == "__main__":
    main()
