import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws

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
    """
    def __init__(self, psi, beta, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, dtype=tf.float64, name = 'JKOSolver'):
        self.psi = psi
        self.beta = beta
        self.ens_file = ens_file
        self.cost_file = cost_file
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.current_time = 0
        super().__init__(name=name, dtype=dtype)

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
            

    def learn_distribution(self, ensemble, weights, epochs=10, initial_rate=1e-3):
        """
        Description:
            attempts to learn the initial condition with Wasserstein_2 loss using the initial ensemble
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
        """
        weights /= weights.sum() 
        self.curr_ref_pts = tf.convert_to_tensor(ensemble, dtype=self.dtype)
        self.prev_weights = tf.convert_to_tensor(weights, dtype=self.dtype)
        cost = ws.compute_cost_matrix(self.curr_ref_pts.numpy(), self.curr_ref_pts.numpy(), p=2)
        self.curr_cost = tf.convert_to_tensor(cost, dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        #print(self.curr_cost)
        #print(self.curr_ref_pts)
        print(True in tf.math.is_nan(tf.reshape(self.curr_cost, (-1,))))
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
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