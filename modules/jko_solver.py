import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws
import vegas
import os
import copy
import utility as ut

class Normalizer(tf.keras.layers.Layer):
    """
    Description:
        A non-trainable layer containing only a scalar
    Args:
        dtype: tf.float32 or tf.float64
    """
    def __init__(self, value=1.0, dtype=tf.float32):
        super().__init__(dtype=dtype, name='Normalizer')
        self.value = value
    
    def build(self, input_shape):
        self.c = self.add_weight(shape=(input_shape[-1], 1), initializer=tf.keras.initializers.Constant(value=self.value), trainable=False,\
                                 name = 'normalization_constant')

    def call(self, x):
        return x / self.c


class JKOSolver(tf.keras.models.Model):
    """
    Description:
        Fokker-Planck solver that uses the JKO method discussed in THE VARIATIONAL FORMULATION OF THE FOKKER–PLANCK EQUATION.
        Solves FP equation of the form rho_t = div((grad psi) * rho) + (Laplacian rho) / beta
    Args:
        psi: potential that is a function of space
        beta: scalar whose inverse denotes the "intensity" of noise
        ens_file: path to ensemble evolution file
        cost_file: path to cost file associated with the ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        dtype: tf.float32 or tf.float64
        name: name of the JKOSolver network
        save_path: path to the directory where a new folder of the same name as the network will be created for storing weights and biases 
    """
    def __init__(self, psi, beta, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, dtype=tf.float32, name = 'JKOSolver', save_path=None):
        self.psi = psi
        self.beta = beta
        self.ens_file = ens_file
        self.cost_file = cost_file
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        hdf5 = tables.open_file(ens_file, 'r')
        self.final_available_time_id, self.time_step, self.ensemble_size, self.dim = hdf5.root.config.read()[0]
        hdf5.close()
        super().__init__(name=name, dtype=dtype)
        self.current_time = 0
        self.normalizer = Normalizer(value=1.0, dtype=self.dtype)
        self.folder = '{}/'.format(save_path) if save_path is not None else '' + '{}'.format(self.name)
        try:
            os.mkdir(self.folder)
        except:
            pass

    def prepare(self):
        """
        Description:
            prepares for the next time step
        """
        # update the clock
        self.current_time += 1
        # set the previous reference points
        self.prev_ref_pts = self.curr_ref_pts
        self.prev_weights = tf.reshape(self.call(self.prev_ref_pts), (-1,))
        self.prev_weights /= tf.reduce_sum(self.prev_weights)
        # set the new reference points
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        pts = getattr(hdf5_ens.root.ensemble, 'time_' + str(self.current_time)).read()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        # set the new cost matrix
        hdf5_cost = tables.open_file(self.cost_file, 'r')
        cost = getattr(hdf5_cost.root, 'time_' + str(self.current_time - 1)).read()
        hdf5_cost.close()
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)

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
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss_W = ws.sinkhorn_loss(self.curr_ref_pts, self.prev_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                loss_E = self.loss_E(self.curr_ref_pts)
                loss_S = self.loss_S(self.curr_ref_pts)
                loss = 0.5 * loss_W + self.time_step * (loss_E + loss_S)
                print('epoch = {}, Wasserstein-loss = {:.4f}, E-loss = {:4f}, S-loss = {:4f}'.format(epoch + 1, loss_W, loss_E, loss_S))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain, nitn=nitn, neval=neval)

    @ut.timer
    def solve(self, domain, final_time_id=None, epochs_per_step=100, initial_rate=1e-3, nitn=10, neval=200):
        """
        Description:
            solves the equation till a given number of time step
        Args:
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
        self.save_weights(time_id='random')
        self.learn_initial_condition(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
        self.save_weights()
        for _ in range(final_time_id):
            self.prepare()
            self.load_weights(time_id='random')
            self.update(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
            self.save_weights()
            
    def loss_E(self, x):
        """
        Description:
            computes integral psi * rho
        Args:
            x: points which are to be used for Monte Carlo evaluation of the integral
            Monte carlo approximation of the integral
        """
        y = tf.py_function(self.psi, inp=[x], Tout=self.dtype)
        return tf.reduce_mean(y)

    def loss_S(self, x):
        """
        Description:
            computes integral psi * log(psi)
        Args:
            x: points which are to be used for Monte Carlo evaluation of the integral
        Returns:
            Monte carlo approximation of the integral
        """
        y = tf.math.log(self.call(x))
        return tf.reduce_mean(y) / self.beta

    def loss_W2(self):
        """
        Description:
            computes the Wasserstein_2 loss
        """
        return ws.sinkhorn_loss(self.prev_ref_pts, self.curr_ref_pts, self.prev_weights, self.curr_weights, self.curr_cost)

    @ut.timer
    def learn_initial_condition(self, domain, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
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
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        pts = getattr(hdf5_ens.root.ensemble, 'time_0').read()
        weights = getattr(hdf5_ens.root.probabilities, 'probs_0').read()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        self.prev_weights = tf.convert_to_tensor(weights, dtype=self.dtype) / weights.sum()
        self.curr_cost = tf.convert_to_tensor(ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2), dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                print('epoch = {}, Wasserstein-loss = {}'.format(epoch + 1, loss.numpy()))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Wasserstein loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain, nitn=nitn, neval=neval)
                

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
        weights /= tf.reduce_mean(weights) 
        self.curr_ref_pts = ensemble#tf.convert_to_tensor(ensemble, dtype=self.dtype)
        self.prev_weights = weights#tf.convert_to_tensor(weights, dtype=self.dtype)
        self.curr_cost = tf.convert_to_tensor(ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2), dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                print('epoch = {}, Wasserstein loss = {}'.format(epoch + 1, loss.numpy()))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Wasserstein loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain, nitn=nitn, neval=neval)


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
        normalization_constant = integrator(integrand, nitn=nitn, neval=neval).mean
        self.normalizer.set_weights([np.array([[normalization_constant]])])
        
    def save_weights(self, time_id=None):
        """
        Description:
            saves model weights with a time index
        Args:
            time_id: a tag for differentiating time
        """
        id = time_id if time_id is not None else str(self.current_time)
        super().save_weights(self.folder + '\weights_' + id)

    def load_weights(self, time_id):
        """
        Description:
            loads model weights given a time index if the weight file exists
        """
        weight_file = self.folder + '\weights_' + str(time_id)
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
            return self.call(x)
        setattr(soln, 'dtype', self.dtype)
        setattr(soln, 'name', self.name)
        return soln