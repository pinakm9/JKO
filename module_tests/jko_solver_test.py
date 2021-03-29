# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import jko_solver as jko
import numpy as np
import tensorflow as tf

dtype = tf.float64

# set psi and beta
def psi(x):
    return (x[:, 0]**2 + x[:, 1]**2 - 1.0)**2

beta = 10.0 
ens_file = 'data/sde_evolve_test_2d.h5'
cost_file = 'sde_evolve_test_2d_cost_2.h5'
# create the solver
solver = jko.JKOSolver(psi, beta, None, dtype)
x = tf.constant([[1., 3.], [2., 4.]], dtype=dtype)
y = tf.constant([2., 4.], dtype=dtype)
print(solver.loss_E(x))