# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
print(root)
sys.path.insert(0, root + '/modules')

import tensorflow as tf
import numpy as np
import copy
import tensorflow_probability as tfp
import derivative as dr
import utility as update

beta = 20.0

class DiffOp():
    def __init__(self, f):
        #super().__init__(name='Equation', dtype=dtype)
        self.f = f

    @tf.function
    def __call__(self, t, x, y):
        z = 4.0*(x*x + y*y - 1.0)
        df = dr.Partial82(self.f)
        f, df0, ddf0 = df(1, t, x, y)
        _, df1, ddf1 = df(2, t, x, y)
  
        return x*z*df0 + y*z*df1 + 4.0*(z + 2.0)*f + (ddf0 + ddf1) / beta

mean = np.zeros(2)
delta = 0.5
cov = delta * np.identity(2)

class InitialPDF():
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.c = tf.cast(tf.math.sqrt((2.0 * np.pi * delta) ** 2), dtype=dtype)
        self.d = tf.cast(delta**2, dtype=dtype)
    def sample(self, size):
        return tf.convert_to_tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=size), dtype=self.dtype)
    def __call__(self, x, y):
        return tf.math.exp(- 0.5 * (x**2 + y**2) / self.d ) / self.c




