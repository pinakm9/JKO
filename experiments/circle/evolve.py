# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
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
import equation as eqn

# create initial ensemle
dtype = tf.float64
np_dtype = np.float64
beta = 20.0
s = np.sqrt(2.0/beta)
num_runs = 10

cov = 0.1 * np.identity(2) 
pdf = eqn.InitialPDF()
initial_ensembles = [pdf.sample(500).numpy() for _ in range(num_runs)]


# create an SDE object
def mu(t, X_t):
    x, y = X_t
    z = 4.0 * (1.0 - x*x - y*y)
    return np.array([x*z, y*z], dtype=np_dtype)

def sigma(t, X_t):
    return s 

steps = 1000
eqn = sde.SDE(2, mu, sigma, 'data/evolution_ic1_{}.h5'.format(steps), dtype=np_dtype)
#eqn.extend(5.0)
# evolve the ensemble and record the evolution2
eqn.evolve(initial_ensembles, 0, 0, 1.0, 1.0/steps)

# animate the evolution2
sde.SDEPlotter('data/evolution_ic1_{}.h5'.format(steps))#, ax_lims=[(-1.5, 1.5), (-1.5, 1.5)])
