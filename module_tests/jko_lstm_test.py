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
import jko_lstm as jko
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import jko_plotter as pltr
import gaussian_circle as gc
import vegas

# set psi and beta
def psi(x):
    return (x[:, 0]**2 + x[:, 1]**2 - 1.0)**2

beta = 10.0 
ens_file = 'data/sde_evolve_test_2d_n.h5'
cost_file = 'sde_evolve_test_2d_cost_2_n.h5'

dtype = tf.float32
dimension = 2
num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)

real_density = CustomDensity()
ensemble = tf.convert_to_tensor(rv.sample(size=200), dtype=dtype)
weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
solver = jko.JKOLSTM(30, 4, psi, beta, ens_file, cost_file, sinkhorn_iters=50)
solver(ensemble)
solver.summary()
solver.load_weights(time_id=2)

class SolverDensity(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=dtype)
        integ = vegas.Integrator(2.5*np.array([[-1.0, 1.0], [-1.0, 1.0]]))
        def integrand(x, n_dim=None, weight=None):
            #print(x, type(x))
            y = solver(tf.convert_to_tensor([x], dtype=tf.float32)).numpy()[0][0]
            #print(y)
            return y
        self.c = integ(integrand, nitn=10, neval=200).mean
        print(self.c)
        
    def call(self, x):
        return solver(x) / self.c


plotter = pltr.JKOPlotter(funcs=[solver, real_density], space=2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]]), num_pts_per_dim=45)
plotter.plot('images/lstm_before.png')
solver.learn_unnormalized_density(ensemble, weights, epochs=350, initial_rate=0.001)
domain = 2.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
solver.compute_normalizer(domain)
#SolverDensity()
plotter = pltr.JKOPlotter(funcs=[solver, real_density], space=2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]]), num_pts_per_dim=45)
plotter.plot('images/lstm_after.png')
solver.save_weights()