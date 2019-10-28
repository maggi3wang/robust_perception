# import nevergrad as ng

# def square(x, y=12):
#     return sum((x - .5)**2) + abs(y)

# optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
# # alternatively, you could use ng.optimizers.registry["OnePlusOne"]
# # (registry is a dict containing all optimizer classes)
# recommendation = optimizer.minimize(square)
# print(recommendation)

# print(sorted(ng.optimizers.registry.keys()))

# import rbfopt
# import numpy as np

# def obj_funct(x):
# 	return x[0]*x[1] - x[2]

# bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] *3),
# 	np.array(['R', 'I', 'R']), obj_funct)

# settings = rbfopt.RbfoptSettings(max_evaluations=50)
# alg = rbfopt.RbfoptAlgorithm(settings, bb)
# val, x, itercount, evalcount, fast_evalcount = alg.optimize()

import cma
from cma.fitness_transformations import EvalParallel2

es = cma.CMAEvolutionStrategy(3 * [1], 1, dict(verbose=-9))

with EvalParallel2(number_of_processes=12) as eval_all:
    while not es.stop():
        X = es.ask()
        es.tell(X, eval_all(X, cma.fitness_functions.elli))
    assert es.result[1] < 1e-13 and es.result[2] < 1500