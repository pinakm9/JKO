# See also https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/real_nvp.py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import jko_solver as jko 
tfb = tfp.bijectors
tfd = tfp.distributions

class JKORealNVP(jko.JKOSolver):

    def __init__(self, output_dim, num_masked, num_bijectors, psi, beta, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, name = 'JKORealNVP'): #** additional arguments for the super class
        dtype = tf.float32
        super().__init__(psi, beta, ens_file, cost_file, sinkhorn_epsilon, sinkhorn_iters, dtype, name)
        self.output_dim = output_dim
        self.num_masked = num_masked
        self.num_bijectors = num_bijectors
        self.shift_and_log_scale_fn = tfb.real_nvp_default_template(hidden_layers=[128, 128])
        # Defining the bijector
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.RealNVP(fraction_masked=0.1, shift_and_log_scale_fn=self.shift_and_log_scale_fn))
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)
    
    def prob(self, x):
        return tf.exp(self.flow.log_prob(x))