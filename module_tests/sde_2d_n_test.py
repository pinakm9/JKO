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
import jko_solver as jkos


# set psi and beta
def psi(x):
    return (x[:, 0]**2 + x[:, 1]**2 - 1.0)**2

beta = 200.0 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

dtype = tf.float32
dimension = 2
num_components = 10
domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)


def diff_op(f, x, y):
    r2 = x*x + y*y
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
        outer_x.watch(x)
        outer_y.watch(y)
        with tf.GradientTape() as inner:
            inner.watch([x, y])
            f_ = f(x, y)
        grad = inner.gradient(f_, [x, y])
        f_x = grad[0]
        f_y = grad[1]
    f_xx = outer_x.gradient(f_x, x)
    f_yy = outer_y.gradient(f_y, y)
    a = (x*z) * f_x
    b = (y*z) * f_y
    c = 4.0 * (z + 2.0) * f_
    return a + b + c + (f_xx + f_yy) / beta

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)

real_density = CustomDensity()
ensemble = tf.convert_to_tensor(rv.sample(size=200), dtype=dtype)
weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
solver = jko.JKOLSTM(50, 4, psi, beta, ens_file, cost_file, sinkhorn_iters=20)
solver(ensemble)
solver.summary()


solver.solve_2(domain, diff_op, 20, 350)
plotter = pltr.JKOPlotter(funcs=[solver], space=domain, num_pts_per_dim=30)
plotter.plot('images/sde_2d_n_sol.png')
#plotter.animate('images/sde_2d_n_sol.mp4', t=[0.0, 0.2], num_frames=24)