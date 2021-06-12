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
import fps7_arch as fp
import equation as eqn

tag = 'ic1_1000'
ens_file = 'data/evolution_{}.h5'.format(tag)
dtype = tf.float64
cov = 0.1 * np.identity(2) 
pdf = eqn.InitialPDF(dtype=dtype)
domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
num_nodes = 20
num_layers = 2
method = 'type7'
arch = 'FPDGM'
solver_name = '{}_{}_{}_{}_{}'.format(arch, num_layers, num_nodes, method, tag)
solver = getattr(fp, arch)(num_nodes, num_layers, eqn.DiffOp, ens_file, domain, pdf, dtype=dtype, name=solver_name)
#solver.summary()
solver.solve(1000, 0, 400)