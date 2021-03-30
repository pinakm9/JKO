# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')
sys.path.insert(0, root + '/custom_dists')

# import required modules
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import jko_plotter as pltr
import gaussian_circle as gc 

dimension = 2
num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)
print(rv.means)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)


density = CustomDensity()
plotter = pltr.JKOPlotter(funcs=[density], space=2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]]), num_pts_per_dim=45)
plotter.plot('images/gaussian_circle_{}.png'.format(num_components))