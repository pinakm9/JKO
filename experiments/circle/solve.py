# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import numpy as np
import tensorflow as tf
import fp_architecture as fp
import equation as eqn

ens_file = 'data/evolution_100.h5'
dtype = tf.float64

domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
solver = fp.FPDGM(20, 3, eqn.Equation, ens_file, domain, eqn.InitialPDF(dtype=dtype), dtype=dtype, name='FPDGM_3_20_t')
solver.summary()
solver.solve(1000, 0, 250)