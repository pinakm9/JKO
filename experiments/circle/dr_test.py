# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import numpy as np
import derivative as dr
import tensorflow as tf
import fp_architecture as fp
import utility as ut
import equation as eqn

ens_file = 'data/evolution.h5'
dtype = tf.float64
beta = 10.0

domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
nn = fp.FPDGM(20, 2, eqn.Equation, ens_file, domain, eqn.InitialPDF())
nn.summary()



def f(t, x):
    return tf.math.exp(-t*x)


@ut.timer
def run(f, input):
    args = tf.split(input, 3, axis=1)
    return f(*args)


@ut.timer
def run_(f, *args):
    return f(*args)

"""
input = tf.convert_to_tensor(np.random.normal(size=(500, 3)))
run(nn.taylor, input)
run(nn.eqn, input)
input = tf.convert_to_tensor(np.random.normal(size=(500, 3)))
run(nn.taylor, input)
run(nn.eqn, input)
"""


"""
h = 0.01

x = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float64)
t = tf.ones_like(x)
fot = dr.FourthOrderTaylor(h)
print(fot(f, t, x))
print(f(t+h, x))

f = lambda t, x: tf.math.sin(t + x)
print(fot(f, t, x))
print(f(t+h, x))
"""
"""
f = lambda t_, x_, y_: tf.math.sin(t_ + x_) * tf.exp(-y_)
op = eqn.Equation(f)
x = tf.constant(np.random.normal(size=(10000, 1)), dtype=tf.float64)
t = tf.ones_like(x)
run_(op, t, x, x)
f = lambda t_, x_, y_: tf.math.sin(t_ + x_) + y_ 
x = tf.constant(np.random.normal(size=(10000, 1)), dtype=tf.float64)
op.f = f
run_(op, t, x, x)
x = tf.constant(np.random.normal(size=(10000, 1)), dtype=tf.float64)
run_(op, t, x, x)
"""
class FourthTaylor():
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
            c = - 4.0 * (z + 2.0)
            f_ = self.f(t, x, y)

            f_x, f_y = tape.gradient(f_, [x, y])
            f_xx = tape.gradient(f_x, x)
            f_yy = tape.gradient(f_y, y)
            Lf_ = a*f_x + b*f_y + c + (f_xx + f_yy - f_x**2 - f_y**2) / beta
          
            #print('Lf_', Lf_)

            Lf_x, Lf_y = tape.gradient(Lf_, [x, y])
            Lf_xx = tape.gradient(Lf_x, x)
            Lf_yy = tape.gradient(Lf_y, y)
            LLf_ = a*Lf_x + b*Lf_y + c + (Lf_xx + Lf_yy - Lf_x**2 - Lf_y**2) / beta
            #print('LLf_', LLf_)

            LLf_x, LLf_y = tape.gradient(LLf_, [x, y])
            LLf_xx = tape.gradient(LLf_x, x)
            LLf_yy = tape.gradient(LLf_y, y)
            LLLf_ = a*LLf_x + b*LLf_y + c + (LLf_xx + LLf_yy - LLf_x**2 - LLf_y**2) / beta
            #print('LLLf_', LLLf_)

            LLLf_x, LLLf_y = tape.gradient(LLLf_, [x, y])
            LLLf_xx = tape.gradient(LLLf_x, x)
            LLLf_yy = tape.gradient(LLLf_y, y)
            LLLLf_ = a*LLLf_x + b*LLLf_y + c + (LLLf_xx + LLLf_yy - LLLf_x**2 - LLLf_y**2) / beta
            #print('LLLf_', LLLf_)

        return f_ + self.h*Lf_ + self.h**2*LLf_/2. + self.h**3*LLLf_/6. + self.h**4*LLLLf_/24.
#"""
h = 0.01         
op = FourthTaylor(nn, h)
x = tf.constant(np.random.normal(size=(1000, 1)), dtype=tf.float64)
t = tf.ones_like(x)
print(run_(op, t, x, x))
x = tf.constant(np.random.normal(size=(1000, 1)), dtype=tf.float64)
t = tf.ones_like(x)
print(run_(op, t, x, x))
x = tf.constant(np.random.normal(size=(1000, 1)), dtype=tf.float64)
t = tf.ones_like(x)
print(run_(op, t, x, x))
#"""
"""
class Test():
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch([x, y])
            f_ = self.f(x, y)
        f_x, f_y = tape.gradient(f_, [x, y]) 
        return f_x, f_y 

f = lambda x, y: tf.exp(-(2*x + y))
g = lambda x, y: x*y 

test = Test(f, g)
x = tf.constant([[0.], [1.]])
y = tf.constant([[1.], [-1.]])
print(test(x, y))
"""
