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
import fps4_arch as fp
import equation as eqn

ens_file = 'data/evolution_gc8_100.h5'
dtype = tf.float64
cov = 0.1 * np.identity(2) 
pdf = eqn.GaussianCircle(cov, np.ones(8))
domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
solver = fp.FPDGM(20, 2, eqn.ThirdSpaceTaylor, eqn.RadialSymmetry, ens_file, domain, pdf, dtype=dtype, name='FPDGM_2_20_type5_gc8_100')
#solver.summary()
solver.solve(2500, 0, 400)