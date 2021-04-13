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
import derivative as dr
import tensorflow as tf
import fp_architecture as fp
import utility as ut
import equation as eqn

ens_file = 'data/evolution.h5'
dtype = tf.float64
beta = 10.0

domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
nn = fp.FPDGM(20, 3, eqn.Equation, ens_file, domain, None)
nn.summary()



@ut.timer
def run(f, input):
    args = tf.split(input, 3, axis=1)
    return f(*args)
input = tf.convert_to_tensor(np.random.normal(size=(500, 3)))
run(nn.taylor, input)
run(nn.eqn, input)
input = tf.convert_to_tensor(np.random.normal(size=(500, 3)))
run(nn.taylor, input)
run(nn.eqn, input)