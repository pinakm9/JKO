import tensorflow as tf 
from contextlib import ExitStack
import math


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

    def call(self, *args):
        with tf.GradientTape() as tape:
            tape.watch(args)
            f = self.func(*args)
        partials = tape.gradient(f, args)
        return partials, f

class SecondPartials(tf.keras.layers.Layer):
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

    def call(self, *args):
        with ExitStack() as stack:
            outer_tapes = [stack.enter_context(tf.GradientTape(persistent=True)) for _ in args]
            for i, arg in enumerate(args):
                outer_tapes[i].watch(arg)
            with tf.GradientTape() as tape:
                tape.watch(args)
                f = self.func(*args)
            first_partials = tape.gradient(f, args)
        second_partials = []
        for partial in first_partials:
            second_partials.append([outer_tapes[i].gradient(partial, arg) for i, arg in enumerate(args)])
        return second_partials, first_partials, f