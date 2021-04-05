import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import jko_solver_nf as jko 
tfb = tfp.bijectors
tfd = tfp.distributions


class SplineParams(tf.Module):

  def __init__(self, nbins=32):
    self._nbins = nbins
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        out_shape = tf.concat((tf.shape(x)[:-1], (nunits, self._nbins)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

      def _slopes(x):
        out_shape = tf.concat((
          tf.shape(x)[:-1], (nunits, self._nbins - 1)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softplus(x) + 1e-2

      self._bin_widths = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='w')
      self._bin_heights = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='h')
      self._knot_slopes = tf.keras.layers.Dense(
          nunits * (self._nbins - 1), activation=_slopes, name='s')
      self._built = True

    return tfb.RationalQuadraticSpline(
        bin_widths=self._bin_widths(x),
        bin_heights=self._bin_heights(x),
        knot_slopes=self._knot_slopes(x))

class JKORQS(jko.JKOSolver):

    def __init__(self, dim, psi, beta, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, name = 'JKORQS'): #** additional arguments for the super class
        dtype = tf.float32
        super().__init__(psi, beta, ens_file, cost_file, sinkhorn_epsilon, sinkhorn_iters, dtype, name)
        self.dim = dim
        #self.num_masked = num_masked
        #self.num_bijectors = num_bijectors
        #self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[500, 500])
        self.splines = [SplineParams() for _ in range(dim)]



        # Defining the bijector
        bijectors=[]
        for i in range(dim):
            bijectors.append(tfb.RealNVP(i, bijector_fn=self.splines[i]))
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

        """
        def spline_flow():
            stack = tfb.Identity()
            for i in range(dim):
                stack = tfb.RealNVP(i, bijector_fn=self.splines[i])(stack)
            return stack
        # Defining the flow
        self.flow = spline_flow
        """
    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)
    
    def prob(self, x):
        return tf.exp(self.flow.log_prob(x))

