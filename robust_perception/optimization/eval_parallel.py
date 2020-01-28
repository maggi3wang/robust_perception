from functools import partial
import numpy as np
from torch.multiprocessing import Pool
import warnings

class EvalParallel3(object):
    def __init__(self, fitness_function=None, number_of_processes=None):
        self.fitness_function = fitness_function
        self.processes = number_of_processes
        self.pool = Pool(self.processes)

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
        self.pool.close()  # would wait for job termination
        # self.pool.terminate()  # terminate jobs regardless
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
