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
import fps4_arch as fp
import utility as ut
import equation as eqn

ens_file = 'data/evolution.h5'
dtype = tf.float64
beta = 1.0

domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
nn = fp.FPDGM(50, 5, eqn.ThirdSpaceTaylor, eqn.RadialSymmetry, ens_file, domain, eqn.InitialPDF())
#nn.summary()



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
class FourthTaylor2():
    def __init__(self, f, h):
        self.f = f
        self.h = h

    #@tf.function
    def __call__(self, *args):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            
            tape.watch(args)
            x, y = args[:2]
            z = 4.0*(x*x + y*y - 1.0)
            a = (x*z - y)
            b = (y*z + x)
            c = 4.0 * (z + 2.0)
            f_ = self.f(*args)

            df_ = tape.gradient(f_, args)
            d2f_ = 0.
            for i in range(len(args)):
                d2f_ += tape.gradient(df_[i], args[i])
            Lf_ = a*df_[0] + b*df_[1] + c*f_ + d2f_ / beta
            print('Lf_', Lf_)
            del df_, d2f_

            dLf_ = tape.gradient(Lf_, args)
            d2Lf_ = 0.
            for i in range(len(args)):
                d2Lf_ += tape.gradient(dLf_[i], args[i])
            LLf_ = a*dLf_[0] + b*dLf_[1] + c*Lf_ + d2Lf_ / beta
            print('LLf_', LLf_)
            del dLf_, d2Lf_

            dLLf_ = tape.gradient(LLf_, args)
        d2LLf_ = 0.
        for i in range(len(args)):
            d2LLf_ += tape.gradient(dLLf_[i], args[i])
        LLLf_ = a*dLLf_[0] + b*dLLf_[1] + c*LLf_ + d2LLf_ / beta
        print('LLLf_', LLLf_)
        del dLLf_, d2LLf_
        return f_ + self.h*Lf_ + self.h**2*LLf_/2. + self.h**3*LLLf_/6. 


#"""
h = 0.01
"""         
op = FourthTaylor2(nn, h)
x = tf.constant(np.random.normal(size=(500, 1)), dtype=tf.float64)
args = [x for _ in range(5)]
print(run_(op, *args))

x = tf.constant(np.random.normal(size=(100, 1)), dtype=tf.float64)
t = tf.ones_like(x)
print(run_(op, *args))
x = tf.constant(np.random.normal(size=(100, 1)), dtype=tf.float64)
t = tf.ones_like(x)
print(run_(op, *args))
#"""



class L:
    def __init__(self, f):
        #super().__init__(name='Equation', dtype=dtype)
        self.f = f
        pass

    @tf.function
    def __call__(self, x, y):
        z = 4.0*(x*x + y*y - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner:
                inner.watch([x, y])
                f_ = self.f(x, y)
            f_x, f_y = inner.gradient(f_, [x, y])
        f_xx = outer_x.gradient(f_x, x)
        f_yy = outer_y.gradient(f_y, y)
        a = (x*z - y) * f_x
        b = (y*z + x) * f_y
        c = 4.0 * (z + 2.0) * f_
        d = (f_xx + f_yy) / beta
        return a + b + c + d
















class Op:
    def __init__(self, f):
        self.f = f 
    @ut.timer
    def __call__(self, *args):
        x, y = args[:2]
        z = 4.0*(x*x + y*y - 1.0)
        a = (x*z - y)
        b = (y*z + x)
        c = 4.0 * (z + 2.0) 
        df =  D8(self.f)
        f, df0, ddf0 = df(0, *args)
        _, df1, ddf1 = df(1, *args)
        _, _, ddf2 = df(2, *args)
        _, _, ddf3 = df(3, *args)
        _, _, ddf4 = df(4, *args)
        _, _, ddf5 = df(5, *args)
        _, _, ddf6 = df(6, *args)
        _, _, ddf7 = df(7, *args)
        _, _, ddf8 = df(8, *args)
        _, _, ddf9 = df(9, *args)
    
  
        return a*df0 + b*df1 + c*f + (ddf0 + ddf1 + ddf2 + ddf3 + ddf4 + ddf5 + ddf6 + ddf7 + ddf8 + ddf9) / beta


class FourthTaylorOp:
    def __init__(self, f):
        self.f0 = f
        self.f1 = Op(self.f0)
        self.f2 = Op(self.f1)
        #self.f3 = Op(self.f2)
        #self.f4 = Op(self.f3)
        self.h = 1e-2
    
    def __call__(self, *args):
        return self.f0(*args) + self.h**1 * self.f1(*args) + self.h**2 * self.f2(*args) / 2. #+ self.h**3 * self.f3(*args) / 6. + self.h**4 * self.f4(*args) / 24.



class dx:
    def __init__(self):
        pass
    def __call__(self, f, *args):
        delta = tf.Variable(0.0, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(delta)
            with tf.GradientTape() as tape2:
                tape2.watch(delta)
                f_ = f(args[0] + delta, *args[1:])
            df_ = tape2.jacobian(f_, delta)
        return tape.jacobian(df_, delta)



class Dy:
    def __init__(self, f):
        self.f = f


    def __call__(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(y)
            f_ = self.f(x, y)
        return tape.gradient(f_, y)

class Op2:
    def __init__(self, f):
        self.f = f


    def __call__(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y)
            f_ = self.f(x, y)
            f_y = tape.gradient(f_, y)
        f_yy = tape.gradient(f_y, y)
        return f_ + f_yy


class Dx:
    def __init__(self, f):
        self.f = f 
        pass
    def __call__(self, i, *args):
        x = args[i]
        h = 1e-2
        a = 0.
        b = 0.

        p = self.f(*args[:i], x-3*h, *args[i+1:])
        a += -p/60.
        b = p/90.

        p = self.f(*args[:i], x-2*h, *args[i+1:])
        a += 3.*p/20.
        b += -3.*p/20. 

        p = self.f(*args[:i], x-h, *args[i+1:])
        a += -3.*p/4.
        b += 3.*p/2.
    
        p = self.f(*args[:i], x+3*h, *args[i+1:])
        a += p/60.
        b += p/90.

        p = self.f(*args[:i], x+2*h, *args[i+1:])
        a += -3.*p/20.
        b += -3.*p/20.

        p = self.f(*args[:i], x+h, *args[i+1:])
        a += 3.*p/4.
        b += 3.*p/2.

        p = self.f(*args[:i], x, *args[i+1:])
        b += -49.*p/18.
        return p, a/h, b/h**2




class D8:
    def __init__(self, f):
        self.f = f 
    
    @ut.timer
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


class ThirdSpaceTaylor():
    def __init__(self, f, h):
        self.f = f
        self.h = h
    
    @tf.function
    def __call__(self, x, y, z0, z1, z2, z3, z4, z5, z6, z7):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            
            tape.watch([x, y, z0, z1, z2, z3, z4, z5, z6, z7])
            z = 4.0*(x*x + y*y - 1.0)
            a = (x*z - y)
            b = (y*z + x)
            c = 4.0 * (z + 2.0)
            f_ = self.f(x, y, z0, z1, z2, z3, z4, z5, z6, z7)

            f_x, f_y, f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7 = tape.gradient(f_, [x, y, z0, z1, z2, z3, z4, z5, z6, z7])
            f_xx = tape.gradient(f_x, x)
            f_yy = tape.gradient(f_y, y)
            f_00 = tape.gradient(f_0, z0)
            f_11 = tape.gradient(f_1, z1)
            f_22 = tape.gradient(f_2, z2)
            f_33 = tape.gradient(f_3, z3)
            f_44 = tape.gradient(f_4, z4)
            f_55 = tape.gradient(f_5, z5)
            f_66 = tape.gradient(f_6, z6)
            f_77 = tape.gradient(f_7, z7)
            Lf_ = a*f_x + b*f_y + c*f_ + (f_xx + f_yy + f_00 + f_11 + f_22 + f_33 + f_44 + f_55 + f_66 + f_77) / beta
          
            print('Lf_', Lf_)

            Lf_x, Lf_y, Lf_0, Lf_1, Lf_2, Lf_3, Lf_4, Lf_5, Lf_6, Lf_7 = tape.gradient(Lf_, [x, y, z0, z1, z2, z3, z4, z5, z6, z7])
            Lf_xx = tape.gradient(Lf_x, x)
            Lf_yy = tape.gradient(Lf_y, y)
            Lf_00 = tape.gradient(Lf_0, z0)
            Lf_11 = tape.gradient(Lf_1, z1)
            Lf_22 = tape.gradient(Lf_2, z2)
            Lf_33 = tape.gradient(Lf_3, z3)
            Lf_44 = tape.gradient(Lf_4, z4)
            Lf_55 = tape.gradient(Lf_5, z5)
            Lf_66 = tape.gradient(Lf_6, z6)
            Lf_77 = tape.gradient(Lf_7, z7)
            LLf_ = a*Lf_x + b*Lf_y + c*Lf_ + (Lf_xx + Lf_yy + Lf_00 + Lf_11 + Lf_22 + Lf_33 + Lf_44 + Lf_55 + Lf_66 + Lf_77) / beta
            print('LLf_', LLf_)
        """
            LLf_x, LLf_y, LLf_0, LLf_1, LLf_2, LLf_3, LLf_4, LLf_5, LLf_6, LLf_7 = tape.gradient(LLf_, [x, y, z0, z1, z2, z3, z4, z5, z6, z7])
        LLf_xx = tape.gradient(LLf_x, x)
        LLf_yy = tape.gradient(LLf_y, y)
        LLf_00 = tape.gradient(LLf_0, z0)
        LLf_11 = tape.gradient(LLf_1, z1)
        LLf_22 = tape.gradient(LLf_2, z2)
        LLf_33 = tape.gradient(LLf_3, z3)
        LLf_44 = tape.gradient(LLf_4, z4)
        LLf_55 = tape.gradient(LLf_5, z5)
        LLf_66 = tape.gradient(LLf_6, z6)
        LLf_77 = tape.gradient(LLf_7, z7)
        LLLf_ = a*LLf_x + b*LLf_y + c*LLf_ + (LLf_xx + LLf_yy + LLf_00 + LLf_11 + LLf_22 + LLf_33 + LLf_44 + LLf_55 + LLf_66 + LLf_77) / beta
        print('LLLf_', LLLf_)
        """
        return f_ + self.h*Lf_ + self.h**2*LLf_/2. #+ self.h**3*LLLf_/6. 


x = tf.constant(np.random.normal(size=(500, 1)), dtype=tf.float64)
args = [x for _ in range(10)]

dr2 = dx()
#print(run_(dr1, nn, *args) - run_(dr2, tf.math.sin, *args))
f = lambda x, y: tf.math.sin(x + y)# + z+ a+ s+ d+ f+ g+ h+ j) #+ z+ t)
g = lambda x, y: tf.math.cos(x + y)# +  z+ a+ s+ d+ f+ g+ h+ j) #+ z+ t)
p = lambda x, y: tf.math.exp(-beta*(x*x + y*y -1.)**2)


cos = Dy(f)
print(cos(x, x) - g(x, x))

zero = Op2(f)
print(zero(x, x))

"""
Lp = L(p)
LLp = L(Lp)
LLLp = L(LLp)
print(run_(LLLp, *args))
"""
#dr1 = Dy(f)
#dr2 = Op(dr1)
#dr3 = Op(dr2)
#print(run_(dr1, x, x) - g(x, x))
#print(run_(dr1, 1, *args)[2] + f(*args))
#"""
"""
dr1 = Op(nn)
dr2 = Op(dr1)
dr3 = Op(dr2)
run_(dr3, *args)
x = tf.constant(np.random.normal(size=(1000, 1)), dtype=tf.float64)
args = [x for _ in range(2)]
run_(dr3, *args)
nn.__summary__()
#"""
"""
d, dd = run_(dr1, 1, *args)
print(d - g(*args))
print(dd + f(*args))
#print(g(*args) + 1)
#print(run_(dr1, 1, *args))
nn.__summary__()
"""
"""
fot_nn = ThirdSpaceTaylor(nn, 1e-2)#FourthTaylorOp(nn)
print(run_(fot_nn, *args))
print(run_(fot_nn, *args))
nn.__summary__()
"""