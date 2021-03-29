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

# create initial ensemle
dtype = np.float64
dimension = 3
mean = np.zeros(dimension)
cov = 0.1 * np.identity(dimension)
initial_ensemble = np.random.random((100, dimension)) * 2 - 1#np.random.multivariate_normal(mean, cov, size=100)

# create an SDE object
def mu(t, X_t):
    x, y, z = X_t
    q = 1.0 - x*x - y*y
    return np.array([x*q - 2.0*y, y*q + 2.0*x, -10.0*z], dtype=dtype)

def sigma(t, X_t):
    return 0.1 * np.identity(dimension, dtype=dtype)

eqn = sde.SDE(dimension, mu, sigma, 'data/sde_evolve_test_3d.h5', dtype=dtype)

# evolve the ensemble and record the evolution
eqn.evolve(initial_ensemble, 5.0, 0.1)

# animate the evolution
sde.SDEPlotter('data/sde_evolve_test_3d.h5', time_step=0.1)