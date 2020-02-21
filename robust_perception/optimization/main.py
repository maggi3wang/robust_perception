from pydrake.math import (RollPitchYaw, RigidTransform)
import numpy as np
import os

from ..optimization.optimizer import OptimizerType
from ..optimization.optimizer import Optimizer
from ..optimization.experiments import Experiment

from ..optimization.model_trainer import MyNet


def train_initial_model():
    """
    Run initial model training
    """

    models_dir = '../data/retrained_with_counterexamples/random1/models'
    training_set_dir = '../data/retrained_with_counterexamples/random1/training_set/initial_training_set'
    test_set_dir = '../data/retrained_with_counterexamples/random1/test_set'
    # counterexample_set_dir = '../data/retrained_with_counterexamples/random1/counterexample_set'

    net = MyNet(
        model_prefix='initial',
        model_trial_number=0,
        num_data_added=0,
        models_dir=models_dir,
        training_set_dirs=[training_set_dir],
        test_set_dir=test_set_dir,
        num_workers=30
    )

    net.train(num_epochs=60)

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

def run_global_optimizers(generate_counterexample_set=False):
    # max_sec = 5 * 60 * 60           # 5 hours
    max_sec = None
    max_counterexamples = None
    max_iterations = None

    # Parameters that are fairly static
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    num_mugs = 3

    package_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(package_directory, '../data/retrained_with_counterexamples/cma_es')

    optimizer = Optimizer(
        num_mugs=num_mugs, mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
        num_processes=30, generate_counterexample_set=generate_counterexample_set,
        retrain_with_counterexamples=True, folder_name=folder_name)
    # optimizer.run_rbfopt()
    optimizer.run_pycma()

def run_random(model_trial_number, generate_counterexample_set=False,
        retrain_with_counterexamples=False, retrain_with_random=False, max_added_to_training=1000):
    """
    generate_counterexample_set: creates a folder with all of the counterexamples found
    retrain_with_counterexamples: retrain model with counterexs
    retrain_with_random: retrain model w random samples
    """

    # Don't let retrain with counterexs and random both be true
    assert(not (retrain_with_counterexamples and retrain_with_random))

    max_sec = None
    max_counterexamples = None
    max_iterations = None

    # Parameters that are fairly static
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    num_mugs = 3

    package_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(package_directory, '../data/retrained_with_counterexamples/random1')

    optimizer = Optimizer(
        num_mugs=num_mugs, mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_added_to_training=max_added_to_training,
        max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
        num_processes=30, generate_counterexample_set=generate_counterexample_set,
        retrain_with_counterexamples=retrain_with_counterexamples,
        retrain_with_random=retrain_with_random, model_trial_number=model_trial_number,
        folder_name=folder_name)
    optimizer.run_random()

def run_random_vs_counterex_experiment():
    # for counterex, rand in zip([False, True], [True, False]):
    #     # Run 10 models by retraining w random sampling, 10 models by retraining w counterexs
    #     for model_trial_number in range(0, 10):
    #         run_random(model_trial_number=model_trial_number, generate_counterexample_set=False,
    #             retrain_with_counterexamples=counterex, retrain_with_random=rand)

    # for counterex, rand in zip([False, True], [True, False]):
    # Run 10 models by retraining w random sampling, 10 models by retraining w counterexs
    for model_trial_number in range(1, 10):
        run_random(model_trial_number=model_trial_number, generate_counterexample_set=False,
            retrain_with_counterexamples=False, retrain_with_random=True, max_added_to_training=1000)
        print('DONE WITH RANDOM TRIAL {}'.format(model_trial_number))

def main():
    """
    Runs entire mug generation, optimization, and retraining pipeline.
    """

    # Find held-out set of counterexamples

    # train_initial_model()

    # run_local_optimizers()

    # run_global_optimizers(True)
    
    # run_global_optimizers()
    run_random_vs_counterex_experiment()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
