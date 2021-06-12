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
import derivative as dr
import equation as eqn


f = lambda t, x, y: tf.math.sin(t + x + y)
g = lambda t, x, y: tf.math.cos(t + x + y)
pdf = lambda t, x, y: tf.exp(-20. * (x*x + y*y - 1.)**2) 

f_ = dr.TimeDerivative42(f)
x = tf.random.normal((10, 1), dtype=tf.float64)
op = eqn.DiffOp(eqn.DiffOp(pdf))
print(f_(x, x, x)[1] - g(x, x, x))
print(op(x, x, x))
