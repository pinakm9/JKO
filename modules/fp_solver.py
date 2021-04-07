import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws
import vegas
import os
import copy
import utility as ut
import scipy.stats as ss
import nn_plotter as pltr
import math

class Normalizer(tf.keras.layers.Layer):
    """
    Description:
        A non-trainable layer containing only a scalar
    Args:
        dtype: tf.float32 or tf.float64
    """
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype=dtype, name='Normalizer')
    
    def build(self, input_shape):
        self.c = self.add_weight(shape=(input_shape[-1], 1), initializer=tf.keras.initializers.Constant(value=1.0), trainable=False,\
                                 name = 'normalization_constant')

    def call(self, x):
        return x / self.c

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



class FPSolver(tf.keras.models.Model):
    """
    Description:
        Solver for Fokker-Planck equation of the form rho_t = L(rho)
    Args:
        diff_op: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        cost_file: path to cost file associated with the ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
        save_path: path to the directory where a new folder of the same name as the network will be created for storing weights and biases 
    """
    def __init__(self, diff_op, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, dtype=tf.float64, name = 'FPSolver', save_path=None,\
                rk_order=2):
        self.ens_file = ens_file
        self.cost_file = cost_file
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        hdf5 = tables.open_file(ens_file, 'r')
        self.final_available_time_id, self.time_step, self.ensemble_size, self.dim = hdf5.root.config.read()[0]
        hdf5.close()
        super().__init__(name=name, dtype=dtype)
        self.current_time = -1
        self.normalizer = Normalizer(dtype=dtype)
        self.diff_op = diff_op
        self.rk_order = rk_order
        self.folder = '{}/'.format(save_path) if save_path is not None else '' + '{}'.format(self.name)
        try:
            os.mkdir(self.folder)
        except:
            pass
        
        

    def summary(self):
        """
        Description:
            builds all layers and displays a summary of the network
        """
        x = tf.zeros((1, 1), dtype=self.dtype)
        args = [x for _ in range(self.dim)]
        self.normalizer(x)
        #self.rk_layer(*args)
        self(tf.concat(args, axis=-1))
        super().summary()
        
    @ut.timer
    def compute_probabilities(self, pts):
        """
        Description:
            computes probabilities associated with the given ensemble using RK method
        """
        args = tf.split(pts, self.dim, axis=1)
        return RKLayer(self.diff_op, self.call_2, self.time_step, self.rk_order, self.dtype)(*args)

    def prepare(self):
        """
        Description:
            prepares for the next time step
        """
        # update the clock
        self.current_time += 1
        hdf5_ens = tables.open_file(self.ens_file, 'r+')
        pts = getattr(hdf5_ens.root.ensemble, 'time_' + str(self.current_time)).read()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        # set the new cost matrix
        hdf5_cost = tables.open_file(self.cost_file, 'r')
        cost = getattr(hdf5_cost.root, 'time_' + str(self.current_time)).read()
        hdf5_cost.close()
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)
        # set the target probabilities
        if self.current_time > 0:
            self.target_weights = tf.reshape(self.compute_probabilities(self.curr_ref_pts), (-1))
            # record the computed probabilities
            if hasattr(hdf5_ens.root.probabilities, 'time_' + str(self.current_time)):
                setattr(hdf5_ens.root.probabilities, 'time_' + str(self.current_time), self.target_weights.numpy())
            else:
                hdf5_ens.create_array(hdf5_ens.root.probabilities, 'time_' + str(self.current_time), self.target_weights.numpy())
        elif self.current_time == 0:
            self.target_weights = tf.convert_to_tensor(getattr(hdf5_ens.root.probabilities, 'time_0').read(), dtype=self.dtype)
        #self.target_weights /= tf.reduce_sum(self.target_weights)
        hdf5_ens.close()


    @ut.timer
    def update(self, domain, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
        """
        Description:
            trains the network to learn the solution at current time step
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
            domain: box domain over which to compute the normalizing constant
            nitn: number of Vegas iterations
            neval: number of function evaluations per iteration
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                #self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss = self.loss_W2()
                print('epoch = {}, Wasserstein-loss = {:.6f}'.format(epoch + 1, loss))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain)
        

    @ut.timer
    def solve(self, domain, final_time_id=None, initial_time_id=0, epochs_per_step=100, initial_rate=1e-3, nitn=10, neval=200):
        """
        Description:
            solves the equation till a given number of time step
        Args:
            initial_time_id: time at which to start solving, represented by an integer
            final_time_id: final time represented as an integer till which the equation is to be solved,
                            should be less than whatever's available in the ensemble evolution file 
            epochs_per_step: number of epochs to train per time step
            initial_rate: initial learning rate
            domain: box domain over which to compute the normalizing constant
            nitn: number of Vegas iterations
            neval: number of function evaluations per iteration
            
        """
        if final_time_id is None:
            final_time_id = self.final_available_time_id
        else:
            final_time_id = min(self.final_available_time_id, final_time_id)
        #self.save_weights(time_id='random')
        plotter = pltr.NNPlotter(funcs=[self], space=domain, num_pts_per_dim=30)
        self.current_time = initial_time_id - 1
        for _ in range(final_time_id + 1):
            self.prepare()
            #self.load_weights(time_id='random')
            self.update(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
            print('prob at 0 = {}'.format(self.call(tf.convert_to_tensor(np.zeros((1, self.dim)), self.dtype))))
            self.save_weights()
            plotter.plot(self.folder + '/time_{}.png'.format(self.current_time), wireframe=True)

    
    def loss_W2(self):
        """
        Description:
            computes the Wasserstein_2 loss
        """
        return ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights,\
                                self.target_weights, self.curr_cost,\
                                epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)

    def loss_Eq(self):
        """
        Description:
            computes the Equality loss
        """
        return tf.reduce_sum(self.curr_weights - self.target_weights)**2

    def loss_DO(self):
        """
        Description:
            computes the differential operator loss
        """
        return tf.reduce_sum(self.curr_weights - self.target_weights)**2


    #@tf.function
    @ut.timer
    def learn_density(self, ensemble, weights, domain, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
        """
        Description:
            attempts to learn the initial condition with Wasserstein_2 loss using the initial ensemble
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
            domain: box domain over which to compute the normalizing constant
            nitn: number of Vegas iterations
            neval: number of function evaluations per iteration
        """
        self.curr_ref_pts = ensemble#tf.convert_to_tensor(ensemble, dtype=self.dtype)
        self.target_weights = weights#tf.convert_to_tensor(weights, dtype=self.dtype)
        self.curr_cost = tf.convert_to_tensor(ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2), dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                loss_W2 = self.loss_W2()
                #loss_Eq = self.loss_Eq()
                loss = loss_W2 #+ loss_Eq
                print('epoch = {}, Wasserstein loss = {:6f}'.format(epoch + 1, loss_W2))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Wasserstein loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain, nitn=nitn, neval=neval)
        #"""
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                #loss_W2 = self.loss_W2()
                loss_Eq = self.loss_Eq()
                loss =  loss_Eq
                print('epoch = {}, Equality loss = {:6f}'.format(epoch + 1, loss_Eq))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Wasserstein loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #"""


    def compute_normalizer(self, domain, nitn=10, neval=200):
        """
        Description:
            calculates normalizing constant using Vegas algorithm

        Args:
            domain: box domain over which to compute the integral
            nitn: number of Vegas iterations
            neval: number of function evaluations per iteration
        """
        integrator = vegas.Integrator(domain)
        def integrand(x, n_dim=None, weight=None):
            return self.call(tf.convert_to_tensor([x], dtype=self.dtype)).numpy()[0][0]
        c = integrator(integrand, nitn=nitn, neval=neval).mean
        self.normalizer.set_weights([self.normalizer.get_weights()[0] * c])
        
    def save_weights(self, time_id=None):
        """
        Description:
            saves model weights with a time index
        Args:
            time_id: a tag for differentiating time
        """
        id = time_id if time_id is not None else str(self.current_time)
        super().save_weights(self.folder + '/weights_' + id)
 

    def load_weights(self, time_id):
        """
        Description:
            loads model weights given a time index if the weight file exists
        """
        weight_file = self.folder + '/weights_' + str(time_id)
        if os.path.isfile(weight_file + '.index'):
            super().load_weights(weight_file).expect_partial()
        else:
            print('Weight file does not exist for time id = {}. Weights were not loaded.'.format(time_id))
            exit()


    def interpolate(self):
        """
        Description:
            interpolates the solution between time steps
        Returns:
            the interpolated function
        """
        def soln(t, x):
            time_id = int(t / self.time_step)
            self.load_weights(time_id)
            print('normalizer = ', self.normalizer)
            return self.call(x)
        setattr(soln, 'dtype', self.dtype)
        setattr(soln, 'name', self.name)
        
        return soln

    def call_2(self, *args):
        x = tf.concat(args, axis=1)
        return self.call(x)