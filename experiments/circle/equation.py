import tensorflow as tf
import numpy as np
import copy

beta = 20.0

class Equation():
    def __init__(self, f):
        #super().__init__(name='Equation', dtype=dtype)
        self.f = f

    @tf.function
    def __call__(self, t, x, y):
        z = 4.0*(x*x + y*y - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([t, x, y])
                f_ = self.f(t, x, y)
            grad = inner.gradient(f_, [t, x, y])
            f_t = grad[0]
            f_x = grad[1]
            f_y = grad[2]
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z - y) * f_x
        b = (y*z + x) * f_y
        c = - 4.0 * (z + 2.0)
        d = (f_xx + f_yy - f_x**2 - f_y**2) / beta
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


