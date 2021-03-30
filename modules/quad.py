#import tensorflow as tf
import numpy as np
import scipy.stats as ss
import tensorflow_probability as tfp

def polar_2(nn, ensemble):
    #transform the ensemble into polar coordinates
    rho = tf.math.sqrt(tf.reduce_sum(ensemble**2, axis=1))
    theta = tf.math.atan2(ensemble[:, 1], ensemble[:, 0])
    polar_ensemble = tf.stack([rho, theta], axis=-1)



def quad_2(func, ensemble):
   return  2.0 * np.pi * np.mean(func(ensemble) * (1.0 + ensemble[:, 0]**2) * (1.0 + ensemble[:, 1]**2))


dim = 2
mean = np.zeros(dim)
cov = np.identity(dim)
ensemble = np.random.multivariate_normal(mean, cov, size=2000)
func = lambda x: ss.multivariate_normal.pdf(x, mean=mean, cov=cov)
print(quad_2(func, ensemble))

print(tfp.monte_carlo.expectation(func, ensemble))
log_func = lambda x: np.log(ss.multivariate_normal.pdf(x, mean=mean, cov=cov))
tfp.mcmc.NoUTurnSampler(log_func, 0.1)


from vegasflow import vegas_wrapper , run_eager
import tensorflow as tf
import vegas
run_eager()
import tensorflow_probability as tfp
tfd = tfp.distributions

def integrand(x, **kwargs):
    """ Function:
       x_{1} * x_{2} ... * x_{n}
       x: array of dimension (events, n)
    """
    return ss.multivariate_normal.pdf(x, mean=mean, cov=cov)

mvn = tfd.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.])

n_dim = 2
n_events = int(1e6)
n_iter = 10
result = vegas_wrapper(integrand, n_dim, n_iter, n_events, compilable=False)
print(result)

integ = vegas.Integrator([[-5, 5], [-5, 5]])

result = integ(integrand, nitn=10, neval=200)
print(result.mean, type(result.mean))