import nevergrad as ng

def square(x, y=12):
    return sum((x - .5)**2) + abs(y)

optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
# alternatively, you could use ng.optimizers.registry["OnePlusOne"]
# (registry is a dict containing all optimizer classes)
recommendation = optimizer.minimize(square)
print(recommendation)