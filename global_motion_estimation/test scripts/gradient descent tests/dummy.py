import numpy as np
from scipy import optimize


def f(x, a): return x**3 - a
def fder(x, a): return 3 * x**2


rng = np.random.default_rng()
x = rng.standard_normal(100)
a = np.arange(-50, 50)
vec_res = optimize.newton(f, x, fprime=fder, args=(a, ), maxiter=200)
print(vec_res)