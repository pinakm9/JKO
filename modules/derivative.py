import tensorflow as tf 
from contextlib import ExitStack
import math
import utility as ut

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