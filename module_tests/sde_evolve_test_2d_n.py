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
beta = 10.0
s = np.sqrt(2.0/beta)
dimension = 2
mean = np.zeros(dimension)
delta = 0.5
cov = delta * np.identity(dimension)


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



pdf = InitialPDF()
d_pdf = dr.FirstPartials(pdf, 2, dtype)
initial_ensemble = pdf.sample(5000)
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
eqn.evolve(initial_ensemble.numpy(), initial_probs, initial_first_partials, 1.0, 0.01)

# animate the evolution
sde.SDEPlotter('data/sde_evolve_test_2d_n_001.h5')#, ax_lims=[(-1.5, 1.5), (-1.5, 1.5)])

# compute the cost matrices
#ws.compute_cost_evolution_fp(ens_file='data/sde_evolve_test_2d_n_001.h5', save_path='data/sde_evolve_test_2d_n_cost_2_001.h5')