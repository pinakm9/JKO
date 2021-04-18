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
import fps3_arch as fp
import equation as eqn
import tables
import nn_plotter as pltr
import scipy.special as ss

class SteadyState(tf.keras.layers.Layer):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta 
        self.c = 0.5 * np.sqrt(np.pi**3 / self.beta) * (1.0 + ss.erf(np.sqrt(self.beta)))
        print(self.c)

    def call(self, x, y):
        return tf.exp(-self.beta * (x**2 + y**2 - 1.0)**2) / self.c


    def plot(self):
        pass

pdf = SteadyState(20.0)
pdf.plot()

plotter = pltr.NNPlotter(funcs=[pdf], space=2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]]), num_pts_per_dim=300)
plotter.plot('data/steady_state.png', wireframe=True) 