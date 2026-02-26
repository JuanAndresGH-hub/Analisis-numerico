import numpy as np

def f(x):
    return 2.5 + 0.8*x**2 - 3.2*x + np.log(x + 1)

# Evaluar en varios puntos
for x in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print(f"f({x}) = {f(x)}")

