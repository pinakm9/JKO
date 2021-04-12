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
import derivative as dr

class Domain:
    def __init__(self, domain, dtype=tf.float64):
        self.dtype = dtype
        self.dim = len(domain)
        self.domain = tf.convert_to_tensor(domain, dtype=dtype)
        

    def sample(self, num_samples, **params):
        samples = []
        for d in range(self.dim):
            samples.append(tf.random.uniform(shape=(num_samples, 1), minval=self.domain[d][0], maxval=self.domain[d][1], dtype=self.dtype))
        self.sample_size =  num_samples
        return samples

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
        self.c = self.add_weight(shape=(input_shape[-1], 1), initializer=tf.keras.initializers.Constant(value=1.0), trainable=True,\
                                 name = 'normalization_constant')

    def call(self, x):
        return x / self.c


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
    def __init__(self, diff_ops, ens_file, domain, init_cond, sinkhorn_epsilon=0.01, sinkhorn_iters=20, dtype=tf.float64,\
                 name = 'FPSolver', save_path=None,
                rk_order=2, num_domain_pts=500):
        super().__init__(name=name, dtype=dtype)
        self.ens_file = ens_file
        self.domain = Domain(domain, dtype=dtype)
        self.init_cond = init_cond
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        hdf5 = tables.open_file(ens_file, 'r')
        self.final_available_time_id, self.time_step, self.ensemble_size, self.dim = hdf5.root.config.read()[0]
        hdf5.close()
        self.current_time = -1
        self.normalizer = Normalizer(dtype=dtype)
        self.diff_op = diff_ops[0](self._log_call)
        self.num_domain_pts = num_domain_pts
        self.num_total_pts = self.ensemble_size + self.num_domain_pts
        self.folder = '{}/'.format(save_path) if save_path is not None else '' + '{}'.format(self.name)
        try:
            os.mkdir(self.folder)
        except:
            pass
        self.rk_layer = dr.RKLayer(diff_ops[1], self.call, self.time_step, 2, self.dtype)


        
    @ut.timer
    def summary(self):
        """
        Description:
            builds all layers and displays a summary of the network
        """
        x = tf.zeros((1, 1), dtype=self.dtype)
        args = [x for _ in range(self.dim + 1)]
        self.normalizer(x)
        self.diff_op(*args)
        self.rk_layer(*args)
        self(*args)
        super().summary()
        

    @ut.timer
    def prepare(self):
        """
        Description:
            prepares for the next time step
        """
        # update the clock
        self.current_time += 1
        hdf5_ens = tables.open_file(self.ens_file, 'r+')
        pts = getattr(hdf5_ens.root.ensemble, 'time_' + str(self.current_time)).read()
        domain_pts = self.domain.sample(self.num_domain_pts)
        self.curr_ref_pts = [tf.concat([np.reshape(pts[:, d], (-1, 1)), domain_pts[d]], axis=0) for d in range(self.dim)]
        if self.current_time > 0:
            self.target_values = self.rk_layer(self.current_time*self.time_step*tf.ones_like(self.curr_ref_pts[0]), *self.curr_ref_pts)
        else:
            self.target_values = self.init_cond(*self.curr_ref_pts)
        hdf5_ens.close()


    @ut.timer
    def update(self, epochs=100, initial_rate=1e-3, nitn=10, neval=200):
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
        if self.current_time == 0:
            self.learn_density(self.domain.domain.numpy(), 2500, initial_rate, nitn, neval,\
                               self.target_values[: 100], 0.0, *[elem[: 100] for elem in self.curr_ref_pts])
            if self.current_time == 0:
                l = int(self.num_total_pts / 2)
                j, k = 0, l
                for _ in range(2):
                    #target_first_partials = [partial[j: k] for partial in self.target_first_partials
                    self.learn_function(1000, initial_rate, self.target_values[j: k], 0.0, *[elem[j: k] for elem in self.curr_ref_pts])
                    j, k = j + l, k + l
        else:
            t = self.current_time * self.time_step
            self.satisfy_diff_op(1000, initial_rate, t, *self.curr_ref_pts)
            self.learn_function(1000, initial_rate, self.target_values, t, *self.curr_ref_pts)
        

    @ut.timer
    def solve(self, final_time_id=None, initial_time_id=0, epochs_per_step=100, initial_rate=1e-3, nitn=10, neval=200):
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
        self.save_weights(time_id='random')
        plotter = pltr.NNPlotter(funcs=[self], space=self.domain.domain.numpy(), num_pts_per_dim=300)
        self.current_time = initial_time_id - 1
        for _ in range(final_time_id + 1):
            self.prepare()
            self.load_weights(time_id='random')
            self.update(epochs=epochs_per_step, initial_rate=initial_rate, nitn=nitn, neval=neval)
            print('learning done for time = {}'.format(self.current_time))
            self.save_weights()
            plotter.plot(self.folder + '/time_{}.png'.format(self.current_time), self.current_time*self.time_step, wireframe=True)

    
    def loss_S2(self, ensemble_1, ensemble_2, curr_values, target_values, cost_matrix):
        """
        Description:
            computes the Sinkhorn_2 loss
        """
        return ws.sinkhorn_loss(ensemble_1, ensemble_2, curr_values, target_values, cost_matrix,\
                                epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iters)

    def loss_Eq_0(self, values):
        """
        Description:
            computes the Equality loss
        """
        return tf.keras.losses.MeanAbsoluteError()(self.values, values)#tf.reduce_mean(tf.math.square(self.values - values))

    def loss_Eq_1(self, first_partials):
        """
        Description:
            computes the Equality loss
        """
        """
        loss = 0.0
        for i, partial in enumerate(first_partials):
            loss +=  tf.reduce_mean(tf.math.square(self.first_partials[i] - partial))
        """
        return tf.keras.losses.MeanSquaredError()(self.first_partials, first_partials)


    def loss_Eq_2(self, second_partials):
        """
        Description:
            computes the Equality loss
        """
        loss = 0.0
        """
        for i in range(self.dim):
            for j in range(i):
                loss +=  tf.reduce_mean(tf.math.square(self.second_partials[i][j] - second_partials[i][j]))
        #"""
        return tf.keras.losses.MeanSquaredError()(self.second_partials, second_partials)

    def loss_KL(self):
        """
        Description:
            computes the differential operator loss
        """
        return tf.reduce_mean(self.target_weights * tf.math.log(self.call(self.curr_ref_pts)/self.target_weights))


    #@tf.function
    @ut.timer
    def learn_density(self, domain, epochs, initial_rate, nitn, neval, weights, t, *args):
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
        ensemble = tf.concat(args, axis=1)
        cost_matrix = tf.convert_to_tensor(ws.compute_cost_matrix(ensemble.numpy(), ensemble.numpy(), p=2), dtype=self.dtype)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        prev_loss = 0.0
        t_ = t * tf.ones_like(args[0])
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.loss_S2(ensemble, ensemble, self.call(t_, *args), weights, cost_matrix)
                print('epoch = {}, Sinkhorn loss = {:6f}'.format(epoch + 1, loss), end='\r')
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of Sinkhorn loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                if tf.math.abs(loss - prev_loss).numpy() < 1e-5:
                    break
                else:
                    prev_loss = loss
        self.compute_normalizer(t, domain, nitn=nitn, neval=neval)

    @ut.timer
    def satisfy_diff_op(self, epochs, initial_rate, t, *args):
        """
        Description: satisfies the differential operator at the current time
        """
        t = t * tf.ones_like(args[0])
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        prev_loss = 0.0
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                eqn, _log_prob = self.diff_op(t, *args)
                loss = tf.reduce_mean(eqn**2)
                print('epoch = {}, DiffOp loss = {:6f}'.format(epoch + 1, loss), end='\r')
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                if tf.math.abs(loss - prev_loss).numpy() < 1e-5:
                    break
                else:
                    prev_loss = loss

    @ut.timer
    def learn_function(self, epochs, initial_rate, values, t, *args):
        """
        Description:
            attempts to learn a function from its values and first partial derivatives
        Args:
            epochs: number of epochs to train
            initial_rate: initial learning rate
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)
        prev_loss = 0.0
        t = t * tf.ones_like(args[0])
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                #self.second_partials, self.first_partials, self.values = self.partials(*tf.split(ensemble, self.dim, axis=1))
                self.values = self.call(t, *args)
                loss =  self.loss_Eq_0(values) #+ self.loss_Eq_1(first_partials) 
                print('epoch = {}, equality loss = {:6f}'.format(epoch + 1, loss), end='\r')
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                if tf.math.abs(loss - prev_loss).numpy() < 1e-9:
                    break
                else:
                    prev_loss = loss


    def compute_normalizer(self, t, domain, nitn=10, neval=200):
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
            x = tf.convert_to_tensor([x], dtype=self.dtype)
            x = tf.split(x, self.dim, axis=1)
            t_ = t * tf.ones_like(x[0])
            return self.call(t_, *x).numpy()[0][0]
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
        def soln(t, *args):
            time_id = int(t / self.time_step)
            self.load_weights(time_id)
            print('normalizer = ', self.normalizer)
            return self.call
        setattr(soln, 'dtype', self.dtype)
        setattr(soln, 'name', self.name)
        
        return soln

    def _log_call(self, *args):
        return -tf.math.log(self.call(*args))