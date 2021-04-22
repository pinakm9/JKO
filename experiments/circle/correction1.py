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
import tensorflow as tf
import fps3_arch as fp
import equation as eqn
import tables
import nn_plotter as pltr
import wasserstein as ws

ens_file = 'data/evolution_1000.h5'
dtype = tf.float64

domain = 2.0*np.array([[-1., 1.], [-1., 1.]])
solver = fp.FPDGM(20, 3, eqn.ThirdSpaceTaylor, eqn.RadialSymmetry, ens_file, domain, eqn.InitialPDF(), name='FPDGM_3_20')
solver.load_weights(time_id=400)
t = 400

hdf5 = tables.open_file(ens_file, 'r')
ensemble = getattr(hdf5.root.ensemble, 'time_' + str(t)).read()
tf_ensemble = tf.split(ensemble, 2, axis=1)
x, y = tf_ensemble
e1 = [-x, y]
e2 = [-x, -y]
e3 = [x, -y] 
tf_ensemble_sym = [tf_ensemble[1], tf_ensemble[0]]
target_values = solver(*tf_ensemble)
sym_target_values = solver(*tf_ensemble_sym)
target_weights = np.ones(500)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


radial = eqn.RadialSymmetry(solver.call)

for epoch in range(500):
    with tf.GradientTape() as tape:
        w = solver(*tf_ensemble)
        l0 = tf.reduce_mean(tf.math.square(w-target_values))
        #l1 = tf.reduce_mean(tf.math.square(solver(*e1)-target_values))
        #l2 = tf.reduce_mean(tf.math.square(solver(*e2)-target_values))
        #l3 = tf.reduce_mean(tf.math.square(solver(*e3)-target_values))
        loss = 0.9*l0 - tf.reduce_mean(tf.math.log(solver(*tf_ensemble))) #+ 0.1*integ
        print('epoch = {}, loss = {:6f}'.format(epoch + 1, loss), end='\r')
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            print('Invalid value encountered during computation of loss. Exiting training loop ...')
            break
        grads = tape.gradient(loss, solver.trainable_weights)
        optimizer.apply_gradients(zip(grads, solver.trainable_weights))
        radial.f = solver.call

plotter = pltr.NNPlotter(funcs=[solver], space=solver.domain.domain.numpy(), num_pts_per_dim=300)
plotter.plot('data/corrected_{}.png'.format(t), wireframe=True)