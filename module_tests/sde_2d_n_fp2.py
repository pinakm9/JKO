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
import tensorflow as tf
import nn_plotter as pltr



beta = 10.0 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

dtype = tf.float64
dimension = 2
delta = 0.5
domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])

class FullDiffOp(tf.keras.layers.Layer):
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
        c = -4.0 * (z + 2.0) 
        d = (f_xx + f_yy - f_x**2 - f_y**2) / beta
        return a + b + c + d - f_t, f_

class SpaceDiffOp(tf.keras.layers.Layer):
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
            f_x = grad[1]
            f_y = grad[2]
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z) * f_x
        b = (y*z) * f_y
        c = 4.0 * (z + 2.0) * f_ 
        d = (f_xx + f_yy) / beta
        return a + b + c + d

class InitialPDF(tf.keras.layers.Layer):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)
        #self.dist = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))
        #self.pdf = self.dist.prob
        self.c = tf.cast(tf.math.sqrt((2.0 * np.pi * delta) ** dimension), dtype=dtype)
        self.d = tf.cast(delta**dimension, dtype=dtype)
    def sample(self, size):
        return tf.convert_to_tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=size), dtype=dtype)
    def call(self, *args):
        x = tf.concat(args, axis=1)
        return tf.math.exp(- 0.5 * tf.reduce_sum(x**2, axis=1, keepdims=True) / self.d ) / self.c

solver = fp.FPDGM(20, 3, [FullDiffOp, SpaceDiffOp], ens_file, domain, InitialPDF().call, sinkhorn_iters=20, sinkhorn_epsilon=0.01,\
                      name='FP2DGM_2d_n', dtype=dtype)
solver.summary()


solver.solve(100, 0, 600, initial_rate=1e-2)
#plotter = pltr.JKOPlotter(funcs=[solver], space=domain, num_pts_per_dim=30)
#plotter.plot('images/sde_2d_n_sol.png')
#plotter.animate('images/sde_2d_n_sol.mp4', t=[0.0, 0.2], num_frames=24)