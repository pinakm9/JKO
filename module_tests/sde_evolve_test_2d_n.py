# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import sde_evolve as sde
import numpy as np
import scipy.stats as ss
import wasserstein as ws
import derivative as dr
import tensorflow_probability as tfp
import tensorflow as tf

# create initial ensemle
dtype = tf.float64
np_dtype = np.float64
beta = 128.0
s = np.sqrt(2.0/beta)
dimension = 2
mean = np.zeros(dimension)
cov = 0.1 * np.identity(dimension)
initial_ensemble = np.random.multivariate_normal(mean, cov, size=3000)

class InitialPDF(tf.keras.layers.Layer):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)
        self.pdf = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov)).prob
    def call(self, *args):
        print('length', len(args), args[0])
        x = tf.concat(args, axis=1)
        return self.pdf(x)

pdf = InitialPDF()
d_pdf = dr.FirstPartials(pdf, 2, dtype)
initial_first_partials, initial_probs = d_pdf(*tf.split(initial_ensemble, dimension, axis=1))
initial_first_partials = [elem.numpy() for elem in initial_first_partials]
initial_probs = initial_probs.numpy()

# create an SDE object
def mu(t, X_t):
    x, y = X_t
    z = 1.0 - x*x - y*y
    return np.array([x*z - 2.0*y, y*z + 2.0*x], dtype=np_dtype)

def mu_(t, X_t):
    x, y = X_t
    z = 4.0 * (1.0 - x*x - y*y)
    return np.array([x*z , y*z], dtype=np_dtype)

def sigma(t, X_t):
    return s * np.identity(dimension, dtype=np_dtype)

eqn = sde.SDE(dimension, mu_, sigma, 'data/sde_evolve_test_2d_n_001.h5', dtype=np_dtype)

# evolve the ensemble and record the evolution
eqn.evolve(initial_ensemble, initial_probs, initial_first_partials, 1.0, 0.01)

# animate the evolution
sde.SDEPlotter('data/sde_evolve_test_2d_n_001.h5')#, ax_lims=[(-1.5, 1.5), (-1.5, 1.5)])

# compute the cost matrices
#ws.compute_cost_evolution_fp(ens_file='data/sde_evolve_test_2d_n_001.h5', save_path='data/sde_evolve_test_2d_n_cost_2_001.h5')