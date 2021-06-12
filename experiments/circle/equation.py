# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
print(root)
sys.path.insert(0, root + '/modules')



import tensorflow as tf
import numpy as np
import copy
import tensorflow_probability as tfp
import derivative as dr

beta = 20.0

class Equation():
    def __init__(self, f):
        #super().__init__(name='Equation', dtype=dtype)
        self.f = f
        pass

    @tf.function
    def __call__(self, t, x, y):
        z = 4.0*(x*x + y*y - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([t, x, y])
                f_ = self.f(t, x, y)
            f_t, f_x, f_y = inner.gradient(f_, [t, x, y])
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z - y) * f_x
        b = (y*z + x) * f_y
        c = 4.0 * (z + 2.0) * f
        d = (f_xx + f_yy) / beta
        return a + b + c + d - f_t

mean = np.zeros(2)
delta = 0.5
cov = delta * np.identity(2)

class InitialPDF():
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.c = tf.cast(tf.math.sqrt((2.0 * np.pi * delta) ** 2), dtype=dtype)
        self.d = tf.cast(delta**2, dtype=dtype)
    def sample(self, size):
        return tf.convert_to_tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=size), dtype=self.dtype)
    def __call__(self, x, y):
        return tf.math.exp(- 0.5 * (x**2 + y**2) / self.d ) / self.c




class InitialPDF2():
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.num_modes = 9
        self.deltas = [delta/5.0] * self.num_modes
        self.cs = [tf.cast(tf.math.sqrt((2.0 * np.pi * d) ** 2), dtype=dtype) for d in self.deltas]
        self.ds = [tf.cast(d**2, dtype=dtype) for d in self.deltas]
        self.means = 0.5*np.array([[-1., 0.], [0., 0.], [1., 0.], [0., 1.], [0., -1.], [-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    
    def sample(self, size):
        samples = np.zeros((size, 2))
        idx = np.random.choice(self.num_modes, size=size, replace=True)
        identity = np.identity(2)
        for i in range(size):
            samples[i, :] = np.random.multivariate_normal(mean=self.means[idx[i]], cov=self.deltas[idx[i]] * identity, size=1)
        return tf.convert_to_tensor(samples, dtype=self.dtype)

    def __call__(self, x, y):
        prob = 0.
        for i in range(self.num_modes):
            prob += (1./self.num_modes) * tf.math.exp(- 0.5 * ((x-self.means[i][0])**2 + (y-self.means[i][1])**2) / self.ds[i] ) / self.cs[i]
        return prob



class InitialPDF3():
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.num_modes = 2
        self.deltas = [delta/5.0] * self.num_modes
        self.cs = [tf.cast(tf.math.sqrt((2.0 * np.pi * d) ** 2), dtype=dtype) for d in self.deltas]
        self.ds = [tf.cast(d**2, dtype=dtype) for d in self.deltas]
        self.means = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    
    def sample(self, size):
        samples = np.zeros((size, 2))
        idx = np.random.choice(self.num_modes, size=size, replace=True)
        identity = np.identity(2)
        for i in range(size):
            samples[i, :] = np.random.multivariate_normal(mean=self.means[idx[i]], cov=self.deltas[idx[i]] * identity, size=1)
        return tf.convert_to_tensor(samples, dtype=self.dtype)

    def __call__(self, x, y):
        prob = 0.
        for i in range(self.num_modes):
            prob += (1./self.num_modes) * tf.math.exp(- 0.5 * ((x-self.means[i][0])**2 + (y-self.means[i][1])**2) / self.ds[i] ) / self.cs[i]
        return prob



class InitialPDF4():
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.num_modes = 1
        self.deltas = [0.3, 0.5] 
        self.c = tf.cast(2.0 * np.pi * 0.3 * 0.5, dtype=dtype)
        self.means = np.array([[0., 0.5]])
    
    def sample(self, size):
        cov = np.array([[0.3**2, 0.], [0., 0.5**2]])
        samples = np.random.multivariate_normal(mean=[0.0, 0.5], cov=cov, size=size)
        return tf.convert_to_tensor(samples, dtype=self.dtype)

    def __call__(self, x, y):
        return tf.math.exp(- 0.5 * ((x-0.0)**2/0.3**2 + (y-0.5)**2/0.5**2)) / self.c


class GaussianCircle:
    """
    Description:
        creates a multimodal distribution aranged on a circle uniformly using iid Gaussians
    Args:
        mean: mean for each Gaussian distribution
        cov: covarinace matrix for each Gaussian distribution
        weights: a 1d array
    """
    def __init__(self, cov, weights, dtype=tf.float64):
        self.cov = cov 
        self.weights = weights / weights.sum()
        self.num_modes = len(weights)
        self.dim = cov.shape[0]
        self.means = np.zeros((self.num_modes, self.dim))
        angle = 2.0 * np.pi / self.num_modes
        self.tf_probs = []
        scale_tril = tf.linalg.cholesky(cov)
        for i in range(self.num_modes):
            self.means[i, :2] = np.cos(i * angle), np.sin(i * angle)
            self.tf_probs.append(tfp.distributions.MultivariateNormalTriL(loc=self.means[i], scale_tril=scale_tril).prob)
        self.dtype = dtype

    def sample(self, size):
        """
        Description:
            samples from the multimodal distribtion
        Args:
            size: number of samples to be generated
        Returns:
             the generated samples
        """
        samples = np.zeros((size, self.dim))
        idx = np.random.choice(self.num_modes, size=size, replace=True, p=self.weights)
        for i in range(size):
            samples[i, :] = np.random.multivariate_normal(mean=self.means[idx[i]], cov=self.cov, size=1)
        return tf.convert_to_tensor(samples, dtype=self.dtype)

    def __call__(self, x, y):
        """
        Description:
            computes probability for given samples in tensorflow format
        Args:
            x: samples at which probability is to be computed
        Returns:
             the computed probabilities
        """
        probs = 0.0
        x = tf.concat([x, y], axis=1)
        for i in range(self.num_modes):
            probs += self.weights[i] * self.tf_probs[i](x)
        return tf.reshape(probs, (-1, 1))





class ThirdSpaceTaylor():
    def __init__(self, f, h):
        self.f = f
        self.h = h
    
    @tf.function
    def __call__(self, x, y):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            
            tape.watch([x, y])
            z = 4.0*(x*x + y*y - 1.0)
            a = (x*z - y)
            b = (y*z + x)
            c = 4.0 * (z + 2.0)
            f_ = self.f(x, y)

            f_x, f_y = tape.gradient(f_, [x, y])
            f_xx = tape.gradient(f_x, x)
            f_yy = tape.gradient(f_y, y)
            Lf_ = a*f_x + b*f_y + c*f_ + (f_xx + f_yy) / beta
          
            #print('Lf_', Lf_)

            Lf_x, Lf_y = tape.gradient(Lf_, [x, y])
            Lf_xx = tape.gradient(Lf_x, x)
            Lf_yy = tape.gradient(Lf_y, y)
            LLf_ = a*Lf_x + b*Lf_y + c*Lf_ + (Lf_xx + Lf_yy) / beta
            #print('LLf_', LLf_)

            LLf_x, LLf_y = tape.gradient(LLf_, [x, y])
        LLf_xx = tape.gradient(LLf_x, x)
        LLf_yy = tape.gradient(LLf_y, y)
        LLLf_ = a*LLf_x + b*LLf_y + c*LLf_ + (LLf_xx + LLf_yy) / beta
        #print('LLLf_', LLLf_)

        return f_ + self.h*Lf_ + self.h**2*LLf_/2. + self.h**3*LLLf_/6. 



class ThirdOrderTaylor():
    def __init__(self, f, h):
        self.f = f
        self.h = h
    
    @tf.function
    def __call__(self, t, x, y):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            
            tape.watch([x, y])
            z = 4.0*(x*x + y*y - 1.0)
            a = (x*z - y)
            b = (y*z + x)
            c = 4.0 * (z + 2.0)
            f_ = self.f(t, x, y)

            f_x, f_y = tape.gradient(f_, [x, y])
            f_xx = tape.gradient(f_x, x)
            f_yy = tape.gradient(f_y, y)
            Lf_ = a*f_x + b*f_y + c*f_ + (f_xx + f_yy) / beta

            Lf_x, Lf_y = tape.gradient(Lf_, [x, y])
            Lf_xx = tape.gradient(Lf_x, x)
            Lf_yy = tape.gradient(Lf_y, y)
            LLf_ = a*Lf_x + b*Lf_y + c*Lf_ + (Lf_xx + Lf_yy) / beta

            LLf_x, LLf_y = tape.gradient(LLf_, [x, y])
        LLf_xx = tape.gradient(LLf_x, x)
        LLf_yy = tape.gradient(LLf_y, y)
        LLLf_ = a*LLf_x + b*LLf_y + c*LLf_ + (LLf_xx + LLf_yy) / beta

        return f_ + self.h*Lf_ + self.h**2*LLf_/2. + self.h**3*LLLf_/6.












class RadialSymmetry():
    def __init__(self, f):
        self.f = f

    def __call__(self, x, y):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([x, y])
            f_ = self.f(x, y)
        f_x, f_y = tape.gradient(f_, [x, y])
        return y*f_x - x*f_y



class DiffOp():
    def __init__(self, f):
        #super().__init__(name='Equation', dtype=dtype)
        self.f = f

    #@tf.function
    def __call__(self, x, y):
        z = 4.0*(x*x + y*y - 1.0)
        df = dr.Partial82(self.f)
        f, df0, ddf0 = df(0, x, y)
        _, df1, ddf1 = df(1, x, y)
  
        return x*z*df0 + y*z*df1 + 4.0*(z + 2.0)*f + (ddf0 + ddf1) / beta


class Taylor2:
    def __init__(self, f, h):
        self.f = f
        self.h = h
    
    @tf.function
    def __call__(self, *args):
        f_ = self.f(*args)
        f_t = DiffOp(self.f)(*args)
        f_tt = DiffOp(DiffOp(self.f))(*args)
        return f_ + self.h**1 * f_t + self.h**2 * f_tt / 2.
