import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws
import vegas
import os

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
        self.normalizer = 1.0
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
        self.current_time += 1
        self.prev_ref_pts = self.curr_ref_pts
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        pts = getattr(hdf5_ens.root, 'time_' + str(self.current_time)).read().tolist()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        self.curr_weights = self.call(self.curr_ref_pts)
        self.prev_weights = tf.ones_like(self.curr_weights) / self.curr_weights.shape[0]
        hdf5_cost = tables.open_file(self.cost_file, 'r')
        cost = getattr(hdf5_cost.root, 'time_' + str(self.current_time)).read().tolist()
        hdf5_cost.close()
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)

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

    def learn_initial_condition(self, epochs=10, initial_rate=1e-3):
        """
        Description:
            attempts to learn the initial condition with Wasserstein_2 loss using the initial ensemble
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
        """
        hdf5_ens = tables.open_file(self.ens_file, 'r')
        pts = getattr(hdf5_ens.root, 'time_0').read().tolist()
        hdf5_ens.close()
        self.curr_ref_pts = tf.convert_to_tensor(pts, dtype=self.dtype)
        self.prev_weights = tf.ones(len(pts), dtype=self.dtype) / len(pts)
        cost = ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2)
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        #print(self.curr_cost)
        #print(self.curr_ref_pts)
        print(True in tf.math.is_nan(tf.reshape(self.curr_cost, (-1,))))
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                #self.curr_weights /= tf.reduce_sum(self.curr_weights)
                #print(self.curr_weights)
                loss = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                print('epoch = {}, Sinkhorn loss = {}'.format(epoch + 1, loss.numpy()))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Sinkhorn loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
            

    #@tf.function
    def learn_unnormalized_density(self, ensemble, weights, epochs=10, initial_rate=1e-3):
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
        cost = ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2)
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                print('epoch = {}, Sinkhorn loss = {}'.format(epoch + 1, loss.numpy()))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Sinkhorn loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

    #@tf.function
    def learn_density(self, ensemble, weights, epochs=10, initial_rate=1e-3):
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
        cost = ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2)
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        #print(self.curr_cost)
        #print(self.curr_ref_pts)
        #print(True in tf.math.is_nan(tf.reshape(self.curr_cost, (-1,))))
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.prob(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                #print(self.curr_weights)
                loss = ws.sinkhorn_loss(self.curr_ref_pts, self.curr_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                print('epoch = {}, Sinkhorn loss = {}'.format(epoch + 1, loss.numpy()))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Sinkhorn loss. Exiting training loop ...')
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
        self.normalizer = integrator(integrand, nitn=nitn, neval=neval).mean
        
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
