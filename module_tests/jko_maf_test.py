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
import jko_maf as jko
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import jko_plotter as pltr
import gaussian_circle as gc 

# set psi and beta
def psi(x):
    return (x[:, 0]**2 + x[:, 1]**2 - 1.0)**2

beta = 10.0 
ens_file = 'data/sde_evolve_test_2d_n.h5'
cost_file = 'sde_evolve_test_2d_cost_2_n.h5'

dtype = tf.float32
dimension = 2
num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)

real_density = CustomDensity()
ensemble = tf.convert_to_tensor(rv.sample(size=200), dtype=dtype)
weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
solver = jko.JKOMAF(dimension, 1, 3, psi, beta, ens_file, cost_file, sinkhorn_iters=20)
solver(ensemble)
solver.summary()


class SolverDensity(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=dtype)
     
    def call(self, x):
        return solver.prob(x) #/ self.c
solver_density = SolverDensity()

#"""
domain = 2.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
plotter = pltr.JKOPlotter(funcs=[solver.prob], space=domain, num_pts_per_dim=45)
plotter.plot('images/maf_before.png')

solver.learn_density(ensemble, weights, domain, epochs=35, initial_rate=0.001)

plotter = pltr.JKOPlotter(funcs=[solver.prob], space=domain, num_pts_per_dim=45)
plotter.plot('images/maf_after.png')
#solver.save_weights()
mean = np.zeros(2)
cov = np.identity(2)
samples = np.random.multivariate_normal(mean, cov, 10000)
probs = ss.multivariate_normal.pdf(samples, mean, cov)
print(tf.reduce_mean(solver.prob(samples) / probs))
print(tf.reduce_mean(solver.prob(ensemble) / weights))
#"""