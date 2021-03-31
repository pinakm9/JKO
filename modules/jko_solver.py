import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws
import vegas
import os
import copy


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
        pts = getattr(hdf5_ens.root, 'time_' + str(self.current_time)).read().tolist()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        # set the new cost matrix
        hdf5_cost = tables.open_file(self.cost_file, 'r')
        cost = getattr(hdf5_cost.root, 'time_' + str(self.current_time - 1)).read().tolist()
        hdf5_cost.close()
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)

    def update(self, epochs=100, initial_rate=1e-3):
        """
        Description:
            trains the network to learn the solution at current time step
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss_W = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                loss_E = self.loss_E(self.curr_ref_pts)
                loss_S = self.loss_S(self.curr_ref_pts)
                loss = loss_W + loss_E + loss_S
                print('epoch = {}, Wasserstein-loss = {:.4f}, E-loss = {:4f}, S-loss = {:4f}'.format(epoch + 1, loss_W, loss_E, loss_S))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Wasserstein loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def solve(self, final_time_id=None, epochs_per_step=100, initial_rate=1e-3):
        """
        Description:
            solves the equation till a given number of time step
        Args:
            final_time_id: final time represented as an integer till which the equation is to be solved,
                            should be less than whatever's available in the ensemble evolution file 
            epochs_per_step: number of epochs to train per time step
            initial_rate: initial learning rate
        """
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        self.final_available_time_id = len(list(hdf5_ens.walk_nodes("/", "Table"))) - 1
        hdf5_ens.close()
        if final_time_id is None:
            final_time_id = self.final_available_time_id
        else:
            final_time_id = min(self.final_available_time_id, final_time_id)
        self.learn_initial_condition(epochs=epochs_per_step, initial_rate=initial_rate)
        self.save_weights()
        for _ in range(final_time_id):
            self.prepare()
            self.update(epochs=epochs_per_step, initial_rate=initial_rate)
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
        return tf.reduce_mean(y)

    def loss_W2(self):
        """
        Description:
            computes the Wasserstein_2 loss
        """
        return ws.sinkhorn_loss(self.prev_ref_pts, self.curr_ref_pts, self.prev_weights, self.curr_weights, self.curr_cost)

    def learn_initial_condition(self, epochs=100, initial_rate=1e-3):
        """
        Description:
            attempts to learn the initial condition with Wasserstein_2 loss using the initial ensemble
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
        """
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        pts = getattr(hdf5_ens.root, 'time_0').read().tolist()
        weights = np.array(getattr(hdf5_ens.root, 'probs_0').read().tolist()).flatten()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        self.prev_weights = tf.convert_to_tensor(weights, dtype=self.dtype) / weights.sum()
        self.curr_cost = tf.convert_to_tensor(ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2), dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        print(True in tf.math.is_nan(tf.reshape(self.curr_cost, (-1,))))
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
                

    #@tf.function
    def learn_unnormalized_density(self, ensemble, weights, epochs=100, initial_rate=1e-3):
        """
        Description:
            attempts to learn the initial condition with Wasserstein_2 loss using the initial ensemble
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
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
        
    def save_weights(self):
        """
        Description:
            saves model weights with a time index
        """
        super().save_weights(self.folder + '\weights_' + str(self.current_time))

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
