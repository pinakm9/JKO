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
import tensorflow_probability as tfp
import utility as ut
 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

# create initial ensemle
dtype = tf.float64
np_dtype = np.float64
beta = 10.0
s = np.sqrt(2.0/beta)
dimension = 2
mean = np.zeros(dimension)
delta = 0.5
cov = delta * np.identity(dimension)
initial_ensemble = np.random.multivariate_normal(mean, cov, size=3000)

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
        return tf.math.exp(- 0.5 * tf.reduce_sum(x  ** 2, axis=1, keepdims=True) / self.d ) / self.c

class FinalPDF(tf.keras.layers.Layer):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)
    def call(self, X):
        x, y = tf.split(X, 2, axis=1)        
        return tf.math.exp(-beta*(x*x + y*y - 1.0)**2)


domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])

"""
final_pdf = FinalPDF()
plotter = pltr.NNPlotter(funcs=[final_pdf], space=domain, num_pts_per_dim=300)
plotter.plot('images/final_pdf.png', wireframe=True)
exit()
"""
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
    

solver = fp.FPVanilla(20, 3, DiffOp, ens_file, sinkhorn_iters=20, sinkhorn_epsilon=0.01, dtype=dtype, rk_order=4)
solver.summary()

@ut.timer
def fn(x):
    solver.rk_layer(*tf.split(x, 2, axis=1))


real_density = InitialPDF()
partials_rd = dr.SecondPartials(real_density.call, 2)
ensemble = real_density.sample(size=200)
fn(ensemble)
exit()
##print(ensemble)
weights = real_density.call(ensemble)
#print(real_density(ensemble))
#"""
domain = 2.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
plotter = pltr.NNPlotter(funcs=[real_density], space=domain, num_pts_per_dim=50)
plotter.plot('images/real_density_2.png')

#"""
region = fps.Domain(domain, dtype)

#solver.learn_density(ensemble[:100], weights[:100], domain, epochs=350, initial_rate=0.001)
for _ in range(12):
    ensemble = region.sample(1000)
    #weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
    sp, first_partials, weights = partials_rd(*tf.split(ensemble, 2, axis=1))
    #
    solver.learn_function(ensemble, weights, first_partials, sp, epochs=500, initial_rate=0.001)
    #solver.compute_normalizer(domain)


#"""
ensemble = region.sample(100)
a = tf.reshape(solver(ensemble), (-1))
b = tf.reshape(real_density.call(ensemble), (-1))
c = tf.math.abs(a - b)
print(c)
print(tf.reduce_mean(c))
plotter = pltr.NNPlotter(funcs=[solver], space=domain, num_pts_per_dim=50)
plotter.plot('images/fp_lstm_after.png', wireframe=True)
#"""