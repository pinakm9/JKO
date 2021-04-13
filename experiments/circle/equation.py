import tensorflow as tf
import numpy as np

beta = 10.0

class Equation(tf.keras.layers.Layer):
    def __init__(self, f, dtype=tf.float64):
        super().__init__(name='Equation', dtype=dtype)
        self.f = f

    @tf.function
    def call(self, t, x, y):
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

class InitialPDF(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype=dtype)
        self.c = tf.cast(tf.math.sqrt((2.0 * np.pi * delta) ** 2), dtype=dtype)
        self.d = tf.cast(delta**2, dtype=dtype)
    def sample(self, size):
        return tf.convert_to_tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=size), dtype=self.dtype)
    def call(self, x, y):
        return tf.math.exp(- 0.5 * (x**2 + y**2) / self.d ) / self.c
