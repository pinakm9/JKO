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
import fp_lstm as fp
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import nn_plotter as pltr
import gaussian_circle as gc
import vegas
import fp_solver as fps
import derivative as dr


beta = 200.0 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

dtype = tf.float64
dimension = 2
num_components = 10
domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, f):
        super().__init__(name='DiffOp', dtype=dtype)
        self.f = f
    
    def call(self, x, y):
        r2 = x*x + y*y
        z = 4.0*(r2 - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([x, y])
                f_ = self.f(x, y)
            grad = inner.gradient(f_, [x, y])
            f_x = grad[0]
            f_y = grad[1]
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z) * f_x
        b = (y*z) * f_y
        c = 4.0 * (z + 2.0) * f_
        return a + b + c + (f_xx + f_yy) / beta
    

solver = fp.FPDGM(50, 4, DiffOp, ens_file, domain, None, sinkhorn_iters=10, sinkhorn_epsilon=0.01, dtype=dtype)
solver.summary()


num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)

    def call_2(self, x, y):
        X = tf.concat([x, y], axis=1)
        return rv.prob(X)

class InitialPDF(tf.keras.layers.Layer):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)
        #self.dist = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))
        #self.pdf = self.dist.prob
        self.c = tf.cast(tf.math.sqrt((2.0 * np.pi * delta) ** dimension), dtype=dtype)
        self.d = tf.cast(delta**dimension, dtype=dtype)
    def sample(self, size):
        return tf.convert_to_tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=size), dtype=dtype)
    def call(self, x):
        return tf.math.exp(- 0.5 * tf.reduce_sum(x**2, axis=1, keepdims=True) / self.d ) / self.c

real_density = CustomDensity()
partials_rd = dr.FirstPartials(real_density.call_2, 2)
#"""
domain = 2.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
plotter = pltr.NNPlotter(funcs=[solver], space=domain, num_pts_per_dim=50)
plotter.plot('images/fp_lstm_before.png')

#"""
ensemble = tf.convert_to_tensor(rv.sample(size=100), dtype=dtype)
weights = tf.convert_to_tensor(np.ones(100), dtype=dtype) #rv.pdf(ensemble)
solver.learn_density(ensemble, weights, domain, epochs=200, initial_rate=0.001)
region = fps.Domain(domain, dtype)
for _ in range(50):
    ensemble = region.sample(1000)
    weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
    solver.learn_function(ensemble, weights, epochs=500, initial_rate=0.001)
    #solver.compute_normalizer(domain)


#"""
ensemble = tf.convert_to_tensor(rv.sample(size=100), dtype=dtype)
a = tf.reshape(solver(ensemble), (-1))
b = rv.pdf(ensemble.numpy())
c = tf.math.abs(a - b)
print(c)
print(tf.reduce_mean(c))
plotter = pltr.NNPlotter(funcs=[solver], space=domain, num_pts_per_dim=50)
plotter.plot('images/fp_lstm_after.png', wireframe=True)
#"""