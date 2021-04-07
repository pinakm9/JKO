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

# create initial ensemle
dtype = np.float32
beta = 200.0
s = 0.1#np.sqrt(2.0/beta)
dimension = 2
mean = np.zeros(dimension)
cov = 0.1 * np.identity(dimension)
initial_ensemble = np.random.multivariate_normal(mean, cov, size=200)
initial_probs = ss.multivariate_normal.pdf(initial_ensemble, mean=mean, cov=cov)
#print(initial_ensemble)
# create an SDE object
def mu(t, X_t):
    x, y = X_t
    z = 1.0 - x*x - y*y
    return np.array([x*z - 2.0*y, y*z + 2.0*x], dtype=dtype)

def mu_(t, X_t):
    x, y = X_t
    z = 4.0 * (1.0 - x*x - y*y)
    return np.array([x*z , y*z], dtype=dtype)

def sigma(t, X_t):
    return s * np.identity(dimension, dtype=dtype)

eqn = sde.SDE(dimension, mu_, sigma, 'data/sde_evolve_test_2d_n_001.h5', dtype=dtype)

# evolve the ensemble and record the evolution
eqn.evolve(initial_ensemble, initial_probs, 1.0, 0.01)

# animate the evolution
sde.SDEPlotter('data/sde_evolve_test_2d_n_001.h5')#, ax_lims=[(-1.5, 1.5), (-1.5, 1.5)])

# compute the cost matrices
ws.compute_cost_evolution_fp(ens_file='data/sde_evolve_test_2d_n_001.h5', save_path='data/sde_evolve_test_2d_n_cost_2_001.h5')