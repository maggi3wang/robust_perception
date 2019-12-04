from pydrake.math import (RollPitchYaw, RigidTransform)
import numpy as np

from ..optimization.optimizer import OptimizerType
from ..optimization.optimizer import Optimizer
from ..optimization.experiments import Experiment

from ..optimization.model_trainer import MyNet


def train_initial_model():
    """
    Run initial model training
    """

    models_dir = '../data/experiment1/models'
    training_set_dir = '../data/experiment1/training_set'
    test_set_dir = '../data/experiment1/test_set'
    counterexample_set_dir = '../data/experiment1/counterexample_set'

    net = MyNet(model_file_number=0, models_dir=models_dir, training_set_dir=training_set_dir,
        test_set_dir=test_set_dir, counterexample_set_dir=counterexample_set_dir)
    net.train(num_epochs=50)

def run_landscape_experiment():
    experiment = Experiment(num_mugs=3)
    experiment.run_experiment()

def run_local_optimizers():
    # Run the two local optimizers, one after the other, along with a random sample method
    # For 3 mugs only using 1 process

    max_sec = 5 * 60 * 60           # 5 hours
    max_counterexamples = None
    max_iterations = None

    # Parameters that are fairly static
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    mug_initial_poses = []
    num_mugs = 3
    for i in range(num_mugs):
        mug_initial_poses += \
            RollPitchYaw(np.random.uniform(0.0, 2.0*np.pi, size=3)).ToQuaternion().wxyz().tolist() + \
            [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2)]

    # slsqp_optimizer = Optimizer(
    #     num_mugs=num_mugs, mug_initial_poses=mug_initial_poses,
    #     mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
    #     max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
    #     num_processes=1, retrain_with_counterexamples=False)

    # # Run optimizer based on optimizer type
    # slsqp_optimizer.run_local_optimizer(local_optimizer_method=OptimizerType.SLSQP, use_input_initial_poses=True)

    # TODO make a reinitialize function
    nelder_mead_optimizer = Optimizer(
        num_mugs=num_mugs, mug_initial_poses=mug_initial_poses,
        mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
        num_processes=1, retrain_with_counterexamples=False)    

    nelder_mead_optimizer.run_local_optimizer(local_optimizer_method=OptimizerType.NELDER_MEAD, use_input_initial_poses=True)

    # optimizer.run_slsqp(use_input_initial_poses=True)
    # optimizer.plot_graphs()

    # optimizer.run_nelder_mead(use_input_initial_poses=True)
    # optimizer.plot_graphs()

    # optimizer.run_random_sample()
    # optimizer.plot_graphs()

def run_global_optimizers():
    max_sec = 5 * 60 * 60           # 5 hours
    max_counterexamples = None
    max_iterations = None

    # Parameters that are fairly static
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    num_mugs = 3
    
    optimizer = Optimizer(
        num_mugs=num_mugs, 
        mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
        num_processes=1, retrain_with_counterexamples=False)
    optimizer.run_rbfopt()


def main():
    """
    Runs entire mug generation, optimization, and retraining pipeline.
    """

    # Find held-out set of counterexamples

    # train_initial_model()

    # run_local_optimizers()
    run_global_optimizers()

if __name__ == "__main__":
    main()
