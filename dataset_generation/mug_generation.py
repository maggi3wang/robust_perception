import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import FrameId
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere
)
from pydrake.geometry.render import CameraProperties, DepthCameraProperties

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
from pydrake.systems.framework import AbstractValue, DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle
import pydrake.systems.sensors as mut

def main():
    np.random.seed(42)
    random.seed(42)
    for scene_iter in range(1):
        try:
            builder = DiagramBuilder()

            # Create a multibody plant and scene graph
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.001))

            # Add ground
            world_body = mbp.world_body()
            ground_shape = Box(2.0, 2.0, 2.0)
            ground_body = mbp.AddRigidBody("ground", SpatialInertia(
                mass=10.0, p_PScm_E=np.array([0.0, 0.0, 0.0]),
                G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
            mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                           RigidTransform(Isometry3(rotation=np.eye(3), translation=[0, 0, -1])))
            mbp.RegisterVisualGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
                np.array([0.5, 0.5, 0.5, 1.]))
            mbp.RegisterCollisionGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
                CoulombFriction(0.9, 0.8))

            # Parses SDF and URDF input files into a MultibodyPlant and (optionally) a SceneGraph.
            parser = Parser(mbp, scene_graph)

            candidate_model_files = [
                "/Users/maggiewang/Workspace/RobotLocomotion/robust_perception/dataset_generation/mug_clean/mug.urdf",
            ]

            n_objects = np.random.randint(3, 7)
            poses = []  # [quat, pos]
            classes = []
            for k in range(n_objects):
                model_name = "model_%d" % k
                model_ind = np.random.randint(0, len(candidate_model_files))
                class_path = candidate_model_files[model_ind]
                classes.append(class_path)
                parser.AddModelFromFile(class_path, model_name=model_name)
                poses.append([
                    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                    [np.random.uniform(-0.1, 0.1),
                     np.random.uniform(-0.1, 0.1),
                     np.random.uniform(0.1, 0.2)]])

            mbp.AddForceElement(UniformGravityFieldElement())
            mbp.Finalize()

            # visualizer = builder.AddSystem(MeshcatVisualizer(
            #     scene_graph,
            #     zmq_url="tcp://127.0.0.1:6000",
            #     draw_period=0.001))
            # builder.Connect(scene_graph.get_pose_bundle_output_port(),
            #                 visualizer.get_input_port(0))

            # diagram = builder.Build()

            # Use HDTV size.
            width = 1280
            height = 720

            color_properties = CameraProperties(
                width=width, height=height, fov_y=np.pi/6,
                renderer_name="renderer")
            depth_properties = DepthCameraProperties(
                width=width, height=height, fov_y=np.pi/6,
                renderer_name="renderer", z_near=0.1, z_far=5.5)

            # Put it at the origin.
            X_WB = RigidTransform()
            # This id would fail if we tried to render; no such id exists.
            parent_id = FrameId.get_new_id()
            camera_poses = mut.RgbdSensor.CameraPoses(
                X_BC=RigidTransform(), X_BD=RigidTransform())
            sensor = mut.RgbdSensor(parent_id=parent_id, X_PB=X_WB,
                                    color_properties=color_properties,
                                    depth_properties=depth_properties,
                                    camera_poses=camera_poses,
                                    show_window=True)

            visualizer = builder.AddSystem(sensor)
            # OutputPort src, InputPort dest
            builder.Connect(scene_graph.get_query_output_port(),
                visualizer.query_object_input_port())
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
            simulator.set_target_realtime_rate(1.0)
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
                query_output = scene_graph.get_query_output_port().Eval(sg_context)
                context = visualizer.CreateDefaultContext()
                # context.FixInputPort(0, AbstractValue.Make(query_output))
                # print(query_output)
                visualizer.Publish(context)

            prog.AddVisualizationCallback(vis_callback, q_dec)
            prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

            ik.AddMinimumDistanceConstraint(0.001, threshold_distance=1.0)

            prog.SetInitialGuess(q_dec, q0)
            print("Solving")
#            print "Initial guess: ", q0
            start_time = time.time()
            solver = SnoptSolver()
            sid = solver.solver_type()
            # SNOPT
            prog.SetSolverOption(sid, "Print file", "test.snopt")
            prog.SetSolverOption(sid, "Major feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Major optimality tolerance", 1e-2)
            prog.SetSolverOption(sid, "Minor feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Scale option", 0)

            print("Solver opts: ", prog.GetSolverOptions(solver.solver_type()))
            print(type(prog))
            result = mp.Solve(prog)
            print("Solve info: ", result)
            print("Solved in %f seconds" % (time.time() - start_time))

            print(result.get_solver_id().name())
            q0_proj = result.GetSolution(q_dec)

            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            simulator.StepTo(10.0)
            q0_final = mbp.GetPositions(mbp_context).copy()

            # sensor_context = sensor.CreateDefaultContext()
            # output = sensor.AllocateOutput()
            # sensor.CalcOutput(sensor_context, output)

            # Get images from computed output.
            # color_index = sensor.color_image_output_port().get_index()
            # print('color_index: ', color_index)
            # color_image = output.get_data(color_index).get_value()
            # print('color_image: ', color_image)
            # color_array = color_image.data
            # print('color_array: ', color_array)

            # hello = color_array
            # print('hello', hello)

            # depth_index = sensor.depth_image_output_port().get_index()
            # depth_image = output.get_data(depth_index).get_value()
            # depth_array = depth_image.data

            # # Show camera info and images.
            # print("Intrinsics:\n{}".format(sensor.depth_camera_info().intrinsic_matrix()))
            # dpi = mpl.rcParams['figure.dpi']
            # figsize = np.array([color_image.width(), color_image.height()*2]) / dpi
            # plt.figure(1, figsize=figsize)
            # plt.subplot(2, 1, 1)
            # plt.imshow(color_array)
            # # plt.subplot(2, 1, 2)
            # # mpl does not like singleton dimensions for single-channel images.
            # # plt.imshow(np.squeeze(depth_array))
            # plt.show()

            time.sleep(1.0)

        except Exception as e:
            print("Unhandled exception ", e)

        except:
            print("Unhandled unnamed exception, probably sim error")

if __name__ == "__main__":
    main()