# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')
sys.path.insert(0, root + '/custom_dists')
# import required modules
import fp_architecture as fp
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import nn_plotter as pltr
import gaussian_circle as gc
import vegas
import fps2 as fps
import derivative as dr


beta = 200.0 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

dtype = tf.float64
dimension = 2
num_components = 10
domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])
t = 42.0

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, f):
        super().__init__(name='DiffOp', dtype=dtype)
        self.f = f
    
    @tf.function
    def call(self, t, x, y):
        r2 = x*x + y*y
        z = 4.0*(r2 - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([t, x, y])
                f_ = self.f(t, x, y)
            grad = inner.gradient(f_, [t, x, y])
            f_t = grad[0]
            f_x = grad[1]
            f_y = grad[2]
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z) * f_x
        b = (y*z) * f_y
        c = 4.0 * (z + 2.0) * f_
        return a + b + c + (f_xx + f_yy) / beta - f_t
    

num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)

    def call(self, x, y):
        X = tf.concat([x, y], axis=1)
        return rv.prob(X)

real_density = CustomDensity()
solver = fp.FPForget(20, 2, DiffOp, ens_file, domain, real_density.call, sinkhorn_iters=20, sinkhorn_epsilon=0.01, dtype=dtype)
solver.summary()

#"""
domain = 2.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
plotter = pltr.NNPlotter(funcs=[solver], space=domain, num_pts_per_dim=50)
plotter.plot('images/fps2_before.png', t)

#"""
ensemble = tf.convert_to_tensor(rv.sample(size=100), dtype=dtype)
weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
solver.learn_density(domain, 1500, 0.001, 10, 200, weights, t, *tf.split(ensemble, dimension, axis=1))

for _ in range(5):
    ensemble = tf.convert_to_tensor(rv.sample(size=1000), dtype=dtype)
    weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
    solver.learn_function(1000, 0.001, weights, t, *tf.split(ensemble, dimension, axis=1))
    #solver.compute_normalizer(domain)


#"""
ensemble = tf.convert_to_tensor(rv.sample(size=100), dtype=dtype)
t_ = tf.convert_to_tensor(t * np.ones((100, 1)), dtype=dtype)
a = tf.reshape(solver(t_, *tf.split(ensemble, dimension, axis=1)), (-1))
b = rv.pdf(ensemble.numpy())
c = tf.math.abs(a - b)
print(c)
print(tf.reduce_mean(c))
plotter = pltr.NNPlotter(funcs=[solver], space=domain, num_pts_per_dim=50)
plotter.plot('images/fps2_after.png', t, wireframe=True)
#"""