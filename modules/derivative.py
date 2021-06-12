import tensorflow as tf 
from contextlib import ExitStack
import math
import utility as ut
import numpy as np

class RKLayer(tf.keras.layers.Layer):
    """
    Description:
        RK4 for y_t = f(y) where f is a linear operator
    Args:
        f: a layer representing f in the ODE
        y: a callable object representing y in the ODE
        step: step size in RK method 
        order: order of RK method
    """
    def __init__(self, f, y, step, order=2, dtype=tf.float64):
        super().__init__(dtype=dtype, name='RK4Layer')
        self.step = step
        self.terms = [y]
        for _ in range(order):
            self.terms.append(f(self.terms[-1]))
          
    
    def call(self, *args):
        z = 0.0
        for i, term in enumerate(self.terms):
            z += self.step**i * term(*args) / math.factorial(i) 
        return z



class FirstPartials(tf.keras.layers.Layer):
    """
    Description:
        Computes first partial derivatives of a given function
    Args:
        func: function to differentiate
        dim: input dimension of the function
    """
    def __init__(self, func, dim, dtype=tf.float64):
        self.func = func
        self.dim = dim
        super().__init__(name='FirstPartials', dtype=dtype)

    @tf.function
    def call(self, *args):
        with tf.GradientTape() as tape:
            tape.watch(args)
            f = self.func(*args)
        partials = tape.gradient(f, args)
        return partials, f

class SecondPartials(tf.keras.layers.Layer):
    """
    Description:
        Computes the second partial derivatives of a given function
    Args:
        func: function to differentiate
        dim: input dimension of the function
    """
    def __init__(self, func, dim, dtype=tf.float64):
        self.func = func
        self.dim = dim
        super().__init__(name='FirstPartials', dtype=dtype)

    @tf.function
    def call(self, *args):
        with ExitStack() as stack:
            outer_tapes = [stack.enter_context(tf.GradientTape(persistent=True)) for _ in args]
            for i, arg in enumerate(args):
                outer_tapes[i].watch(arg)
            with tf.GradientTape() as tape:
                tape.watch(args)
                f = self.func(*args)
            first_partials = tape.gradient(f, args)
        second_partials = [outer_tapes[j].gradient(partial, args[j]) for j in range(i) for i, partial in enumerate(first_partials)]
        return second_partials, first_partials, f

class DiagSecondPartials(tf.keras.layers.Layer):
    """
    Description:
        Computes the diagonal second partial derivatives of a given function
    Args:
        func: function to differentiate
        dim: input dimension of the function
    """
    def __init__(self, func, dim, dtype=tf.float64):
        self.func = func
        self.dim = dim
        super().__init__(name='FirstPartials', dtype=dtype)

    @tf.function
    def call(self, *args):
        with ExitStack() as stack:
            outer_tapes = [stack.enter_context(tf.GradientTape()) for _ in args]
            for i, arg in enumerate(args):
                outer_tapes[i].watch(arg)
            with tf.GradientTape() as tape:
                tape.watch(args)
                f = self.func(*args)
            first_partials = tape.gradient(f, args)
        second_partials = [outer_tapes[i].gradient(partial, args[i]) for i, partial in enumerate(first_partials)]
        return second_partials, first_partials, f



class FourthOrderTaylor():
    def __init__(self, f, h):
        self.f = f
        self.h = h

    @tf.function
    def __call__(self, t, *args):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(t)
            f_ = self.f(t, *args)
            f_t = tape.gradient(f_, t)
            f_tt = tape.gradient(f_t, t)
            f_ttt = tape.gradient(f_tt, t)
        f_tttt = tape.gradient(f_ttt, t)
        return f_ + self.h*f_t + self.h**2*f_tt/2. + self.h**3*f_ttt/6. + self.h**4*f_tttt/24.


class FifthOrderTaylor():
    def __init__(self, f, h,):
        self.f = f
        self.h = h

    @tf.function
    def __call__(self, t, *args):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(t)
            f_ = self.f(t, *args)
            f_t = tape.gradient(f_, t)
            f_tt = tape.gradient(f_t, t)
            f_ttt = tape.gradient(f_tt, t)
            f_tttt = tape.gradient(f_ttt, t)
        f_ttttt = tape.gradient(f_tttt, t)
        return f_ + self.h*f_t + self.h**2*f_tt/2. + self.h**3*f_ttt/6. + self.h**4*f_tttt/24. + self.h**5*f_ttttt/120. 


class TimeDerivative42:
    def __init__(self, f):
        self.f = f
        self.h = 1e-2
        c = np.array([1./12., -2./3.])
        self.coeff_1 = np.append(c, [0.])
        self.coeff_1 = np.append(self.coeff_1, -c[::-1])
        c = np.array([-1./12., 4./3.])
        self.coeff_2 = np.append(c, [-5./2.])
        self.coeff_2 = np.append(self.coeff_2, -c[::-1])

    @tf.function
    def __call__(self, *args):
        t = args[0]

        a, b, j = 0., 0., 0
        for k in [-2, -1]:
            p = self.f(t + k*self.h, *args[1:])
            a += self.coeff_1[j] * p 
            b += self.coeff_2[j] * p 
            j += 1

        j = 3
        for k in [1, 2]:
            p = self.f(t + k*self.h, *args[1:])
            a += self.coeff_1[j] * p
            b += self.coeff_2[j] * p
            j += 1

        p = self.f(*args)
        b += self.coeff_2[2] * p

        return p, a/self.h, b/self.h**2

class Partial82:
    def __init__(self, f):
        self.f = f 
    
    #@ut.timer
    def __call__(self, i, *args):
        x = args[i]
        h = 1e-2
        a = 0.
        b = 0.

        coeff = np.array([1./280., -4./105., 1./5., -4./5.])
        ac = np.append(coeff, [0])
        ac = np.append(ac, -coeff[::-1])
        

        coeff = np.array([-1./560., 8./315., -1./5., 8./5])
        bc = np.append(coeff, [-205./72.])
        bc = np.append(bc, coeff[::-1])

        j = 0
        for k in range(-4, 0, 1):
            p = self.f(*args[:i], x+k*h, *args[i+1:])
            a += ac[j] * p 
            b += bc[j] * p
            j += 1

        j = 5
        for k in range(1, 5, 1):
            p = self.f(*args[:i], x+k*h, *args[i+1:])
            a += ac[j] * p 
            b += bc[j] * p
            j += 1

        p = self.f(*args) 
        b += bc[4] * p
        
        return p, a/h, b/h**2


class Taylor2:
    def __init__(self, Op, f, h):
        self.f = f
        self.Op = Op
        self.h = h
    
    @tf.function
    def __call__(self, *args):
        f_ = self.f(*args)
        f_t = self.Op(self.f)(*args)
        f_tt = self.Op(self.Op(self.f))(*args)
        return f_ + self.h**1 * f_t + self.h**2 * f_tt / 2., f_t, f_tt