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

pdf = eqn.InitialPDF()
d_pdf = dr.FirstPartials(pdf, 2, dtype)
initial_ensemble = pdf.sample(500)
initial_first_partials, initial_probs = d_pdf(*tf.split(initial_ensemble, 2, axis=1))
initial_first_partials = [elem.numpy() for elem in initial_first_partials]
initial_probs = initial_probs.numpy()

# create an SDE object
def mu(t, X_t):
    x, y = X_t
    z = 4.0 * (1.0 - x*x - y*y)
    return np.array([x*z + y, y*z - x], dtype=np_dtype)

def sigma(t, X_t):
    return s 

steps = 1000
eqn = sde.SDE(2, mu, sigma, 'data/evolution_{}_helper.h5'.format(steps), dtype=np_dtype)

# evolve the ensemble and record the evolution
eqn.evolve(initial_ensemble.numpy(), initial_probs, initial_first_partials, 1.0, 1.0/steps)

# animate the evolution
sde.SDEPlotter('data/evolution_{}_helper.h5'.format(steps))#, ax_lims=[(-1.5, 1.5), (-1.5, 1.5)])
