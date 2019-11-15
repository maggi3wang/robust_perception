from ..optimization.optimizer import OptimizerType
from ..optimization.optimizer import Optimizer

def main():
    """
    Runs entire mug generation, optimization, and retraining pipeline.
    """

    ### Initial parameters

    # Parameters that are fairly variable
    optimizer_type = OptimizerType.PYCMA

    max_sec = None
    max_counterexamples = 100
    max_iterations = None
    # max_sec = 60.0 * 60.0 * 5.0
    # max_counterexamples = 100
    # max_iterations = 50000

    # Parameters that are fairly static
    mug_lower_bound = [-1.0, -1.0, -1.0, -1.0, -0.1, -0.1, 0.1]
    mug_upper_bound = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.2]

    optimizer = Optimizer(
        num_mugs=3, mug_lower_bound=mug_lower_bound, mug_upper_bound=mug_upper_bound,
        max_iterations=max_iterations, max_time=max_sec, max_counterexamples=max_counterexamples,
        num_processes=20, retrain_with_counterexamples=True)

    # Run optimizer based on optimizer type
    if optimizer_type == OptimizerType.PYCMA:
        optimizer.run_pycma()
        optimizer.plot_graphs()
    elif optimizer_type == OptimizerType.RBFOPT:
        optimizer.run_rbfopt()
        optimizer.plot_graphs()
    elif optimizer_type == OptimizerType.NEVERGRAD:
        optimizer.plot_graphs(optimizer.run_nevergrad())
    elif optimizer_type == OptimizerType.SLSQP:
        optimizer.run_scipy_fmin_slsqp()
        optimizer.plot_graphs()
    elif optimizer_type == OptimizerType.NELDER_MEAD:
        optimizer.run_scipy_nelder_mead()
        optimizer.plot_graphs()

if __name__ == "__main__":
    main()
