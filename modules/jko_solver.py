import tensorflow as tf
import numpy as np
import tables
import wasserstein as ws
import vegas
import os
import copy
import utility as ut
import scipy.stats as ss
import jko_plotter as pltr

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
        self.normalizer = 1.0
        self.folder = '{}/'.format(save_path) if save_path is not None else '' + '{}'.format(self.name)
        try:
            os.mkdir(self.folder)
        except:
            pass
        if True:#not os.path.isfile(self.folder + '/normalizer.h5'):
            hdf5 = tables.open_file(self.folder + '/normalizer.h5', 'w')
            hdf5.close()

    def prepare(self):
        """
        Description:
            prepares for the next time step
        """
        # update the clock
        self.current_time += 1
        # set the previous reference points
        self.prev_ref_pts = copy.deepcopy(self.curr_ref_pts)
        self.importance_density = self.call(self.prev_ref_pts)
        self.prev_weights = tf.reshape(self.importance_density, (-1,))
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
                self.compute_normalizer_2(self.prev_ref_pts, self.importance_density)
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss_W = ws.sinkhorn_loss(self.curr_ref_pts, self.prev_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                loss_F = self.loss_F()
                loss = 0.5 * loss_W + self.time_step * loss_F + (tf.reduce_mean(self.call(self.prev_ref_pts) / self.importance_density) - 1.0)**2
                print('epoch = {}, Wasserstein-loss = {:.4f}, F-loss = {:.4f}, total loss = {:.4f}'.format(epoch + 1, loss_W, loss_F, loss))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain)
        

    @ut.timer
    def update_2(self, domain, diff_op, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
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
        x = tf.reshape(self.prev_ref_pts[:, 0], (-1, 1))
        y = tf.reshape(self.prev_ref_pts[:, 1], (-1, 1))
        if self.current_time == 1:
            self.prev_weights = self.call(self.prev_ref_pts) + diff_op(self.call_2, x, y) * self.time_step
            self.prev_weights = tf.reshape(self.prev_weights, (-1,))
            self.prev_weights /= tf.reduce_sum(self.prev_weights)
        else:
            self.load_weights(time_id=self.current_time - 2)
            rho = self.call(self.prev_ref_pts)
            self.load_weights(time_id=self.current_time - 1)
            self.prev_weights = rho + diff_op(self.call_2, x, y) * self.time_step * 2.0
            self.prev_weights = tf.reshape(self.prev_weights, (-1,))
            self.prev_weights /= tf.reduce_sum(self.prev_weights)
        print(self.prev_weights, tf.reduce_sum(self.prev_weights))
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.curr_weights = tf.reshape(self.call(self.curr_ref_pts), (-1,))
                self.curr_weights /= tf.reduce_sum(self.curr_weights)
                loss_W = ws.sinkhorn_loss(self.curr_ref_pts, self.prev_ref_pts, self.curr_weights, self.prev_weights, self.curr_cost,\
                                         epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)
                loss = 0.5 * loss_W 
                print('epoch = {}, Wasserstein-loss = {:.4f}'.format(epoch + 1, loss_W))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compute_normalizer(domain)

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
        #self.save_weights(time_id='random')
        plotter = pltr.JKOPlotter(funcs=[self], space=domain, num_pts_per_dim=30)
        self.learn_initial_condition(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
        self.save_weights()
        plotter.plot(self.folder + '/time_0.png')
        for _ in range(final_time_id):
            self.prepare()
            #self.load_weights(time_id='random')
            self.update(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
            self.save_weights()
            plotter.plot(self.folder + '/time_{}.png'.format(self.current_time))

    @ut.timer
    def solve_2(self, domain, diff_op, final_time_id=None, epochs_per_step=100, initial_rate=1e-3, nitn=10, neval=200):
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
        #self.save_weights(time_id='random')
        plotter = pltr.JKOPlotter(funcs=[self], space=domain, num_pts_per_dim=30)
        self.learn_initial_condition(domain, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
        self.save_weights()
        plotter.plot(self.folder + '/time_0.png')
        for _ in range(final_time_id):
            self.prepare()
            #self.load_weights(time_id='random')
            self.update_2(domain, diff_op, epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
            self.save_weights()
            plotter.plot(self.folder + '/time_{}.png'.format(self.current_time))
            
    def loss_F(self):
        """
        Description:
            computes integral psi * rho + integral psi * log(psi) / beta
        Args:
            x: points which are to be used for Monte Carlo evaluation of the integral
            Monte carlo approximation of the integral
        """
        rho = self.call(self.prev_ref_pts)
        return tf.reduce_mean((self.psi(self.prev_ref_pts) + tf.math.log(rho) / self.beta) * rho / self.importance_density)

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
        #weights_ = copy.deepcopy(weights) 
        weights /= tf.reduce_sum(weights)
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
        """
        mean, cov = np.zeros(2), 2.0*np.identity(2)
        samples = np.random.multivariate_normal(mean, cov, 2000)
        imp = ss.multivariate_normal.pdf(samples, mean, cov)
        samples = tf.convert_to_tensor(samples, dtype=self.dtype)
        imp = tf.convert_to_tensor(imp, dtype=self.dtype)
        
        print('MC integral 1: {}'.format(tf.reduce_mean(tf.divide(self.call(samples),  imp))))
        self.normalizer = tf.reduce_mean(tf.divide(self.call(ensemble),  weights_))
        print('MC integral 2: {}'.format(self.normalizer))
        """
        self.compute_normalizer(domain, nitn=nitn, neval=neval)
        
        print('LV integral: {}'.format(self.normalizer))

     #@tf.function
    @ut.timer
    def learn_density_2(self, ensemble, weights, domain, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
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
        weights_ = copy.deepcopy(weights) 
        weights /= tf.reduce_sum(weights)
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
        self.compute_normalizer(domain, nitn, neval)

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
        self.normalizer *= integrator(integrand, nitn=nitn, neval=neval).mean
        #self.normalizer.set_weights([np.array([[normalization_constant]])])

    def compute_normalizer_2(self, ensemble, importance_density):
        """
        Description:
            calculates normalizing constant using Vegas algorithm

        Args:
            domain: box domain over which to compute the integral
            nitn: number of Vegas iterations
            neval: number of function evaluations per iteration
        """
        mean = np.zeros(2)
        cov = np.identity(2)
        samples = np.random.multivariate_normal(mean, cov, 1000)
        probs = ss.multivariate_normal.pdf(samples, mean, cov)
        self.normalizer *= tf.reduce_mean(self.call(ensemble) / importance_density)
        
    def save_weights(self, time_id=None):
        """
        Description:
            saves model weights with a time index
        Args:
            time_id: a tag for differentiating time
        """
        id = time_id if time_id is not None else str(self.current_time)
        super().save_weights(self.folder + '/weights_' + id)
        hdf5 = tables.open_file(self.folder + '/normalizer.h5', 'a')
        hdf5.create_array(hdf5.root, 'time_' + id, np.array([self.normalizer]))
        hdf5.close()

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
        hdf5 = tables.open_file(self.folder + '/normalizer.h5', 'r')
        self.normalizer = getattr(hdf5.root, 'time_' + str(time_id)).read()[0]
        hdf5.close()

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