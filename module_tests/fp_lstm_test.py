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
import fp_lstm as fp
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import nn_plotter as pltr
import gaussian_circle as gc
import vegas
import fp_solver as fps
import derivative as dr
import utility as ut
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt

beta = 200.0 
ens_file = 'data/sde_evolve_test_2d_n_001.h5'
cost_file = 'data/sde_evolve_test_2d_n_cost_2_001.h5'

dtype = tf.float64
dimension = 2
num_components = 10
domain = 2.0*np.array([[-1.0, 1.0], [-1.0, 1.0]])

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, f):
        super().__init__(name='DiffOp', dtype=dtype)
        self.f = f
    
    def call(self, x, y):
        r2 = x*x + y*y
        z = 4.0*(r2 - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([x, y])
                f_ = self.f(x, y)
            grad = inner.gradient(f_, [x, y])
            f_x = grad[0]
            f_y = grad[1]
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z) * f_x
        b = (y*z) * f_y
        c = 4.0 * (z + 2.0) * f_
        return a + b + c + (f_xx + f_yy) / beta
    

solver = fp.FPDGM(20, 3, DiffOp, ens_file, domain, None, sinkhorn_iters=20, sinkhorn_epsilon=0.01, dtype=dtype)
solver.summary()


num_components = 10
cov = 0.1*np.identity(dimension)
weights = np.ones(num_components)
rv = gc.GaussianCircle(cov, weights)

class CustomDensity(tf.keras.models.Model):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)

    def call(self, x):
        return tf.convert_to_tensor(rv.pdf(x), dtype=self.dtype)

    def call_2(self, x, y):
        X = tf.concat([x, y], axis=1)
        return rv.prob(X)

class CustomDensity2(tf.keras.models.Model):
    def __init__(self, dtype=dtype):
        super().__init__(dtype=dtype)
        self.c = 17.5937 #5.13038
        self.num_burnin_steps = 200
        self.target_log_prob_fn = lambda x: -(tf.reduce_sum(x**2, axis=1) - 1)**2
        sampler = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=self.target_log_prob_fn, step_size=0.1)
        self.adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel=sampler, num_adaptation_steps=int(0.8 * self.num_burnin_steps),\
        target_accept_prob=0.75,\
    # NUTS inside of a TTK requires custom getter/setter functions.
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(inner_results=pkr.inner_results._replace(step_size=new_step_size)\
        ),\
    step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,\
    log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,)



    def call(self, x):
        return tf.exp(-(x[:, 0]**2 + x[:, 1]**2  - 1.0)**2)

    def call_2(self, x, y):
        return tf.exp(-(x**2 + y**2  - 1.0)**2)
    



dims = 10
true_stddev = tf.sqrt(tf.linspace(1., 3., dims))
likelihood = tfd.MultivariateNormalDiag(loc=0., scale_diag=true_stddev)

@ut.timer
@tf.function
def sample(size):
    return tfp.mcmc.sample_chain(\
        num_results=size,\
        num_burnin_steps=1000,\
        current_state=tf.zeros(2),\
        kernel=tfp.mcmc.NoUTurnSampler(\
        target_log_prob_fn=lambda x: -(x[0]**2 + x[1]**2 - 1.0)**2,\
        step_size=0.01),\
        trace_fn=None)

Y = sample(500)
X =Y.numpy()
print(X)
"""
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(111)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
#"""






















real_density = CustomDensity2()

#partials_rd = dr.FirstPartials(real_density.call_2, 2)
#"""
domain = 2.0 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
plotter = pltr.NNPlotter(funcs=[real_density.call_2], space=domain, num_pts_per_dim=300)
plotter.plot('images/real.png')

"""
#ensemble = real_density.sample(200)#tf.convert_to_tensor(rv.sample(size=200), dtype=dtype)
weights =  np.ones(500)#real_density(Y)
print(weights)
solver.learn_density(Y, weights, domain, epochs=200, initial_rate=0.001, p=2)
Y = sample(500)
weights =  np.ones(500)#real_density(Y)
solver.learn_density(Y, weights, domain, epochs=200, initial_rate=0.001, p=2)
#"""
region = fps.Domain(domain, dtype)
for _ in range(2):
    Y = sample(100)
    weights =  real_density(Y)
    #weights = tf.convert_to_tensor(rv.pdf(ensemble), dtype=dtype)
    solver.learn_function(Y, weights, epochs=500, initial_rate=0.001)
    #solver.compute_normalizer(domain)


"""
"""
Y = sample(100)
a = tf.reshape(solver(Y), (-1))
b = real_density(Y).numpy()
c = tf.math.abs(a - b)
print(c)
print(tf.reduce_mean(c))
#"""
plotter = pltr.NNPlotter(funcs=[solver.call_2], space=domain, num_pts_per_dim=300)
plotter.plot('images/fp_lstm_after.png')#, wireframe=True)
#"""