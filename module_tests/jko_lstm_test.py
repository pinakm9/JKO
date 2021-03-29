# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import jko_lstm as jko
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import jko_plotter as pltr

dtype = tf.float32

# set psi and beta
def psi(x):
    return (x[:, 0]**2 + x[:, 1]**2 - 1.0)**2

beta = 10.0 
ens_file = 'data/sde_evolve_test_2d_n.h5'
cost_file = 'sde_evolve_test_2d_cost_2_n.h5'

# create the solver
solver = jko.JKOLSTM(50, 4, psi, beta, ens_file, cost_file, sinkhorn_iters=50)
x = tf.constant([[1., 2.], [3., 4.]], dtype=dtype)
print(solver(x))
solver.summary()

dimension = 2
mean = np.zeros(dimension)
cov = np.identity(dimension)
ensemble = np.random.multivariate_normal(mean, cov, size=200)
weights = ss.multivariate_normal.pdf(ensemble, mean=mean, cov=cov)

class GaussianDensity(tf.keras.models.Model):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        super().__init__()
    def call(self, x):
        y = ss.multivariate_normal.pdf(x.numpy(), mean=self.mean, cov=self.cov)
        return tf.convert_to_tensor(y, dtype=dtype)

real_density = GaussianDensity(mean, cov)
plotter = pltr.JKOPlotter(funcs=[real_density], space=1.0*np.array([[-1.0, 1.0], [-1.0, 1.0]]))
plotter.plot('images/before_initial_training_n.png')
solver.learn_distribution(ensemble, weights, epochs=50, initial_rate=0.001)
plotter.plot('images/after_initial_training_n.png')