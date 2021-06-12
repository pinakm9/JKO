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
        

    def sample(self, num_samples):
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
    def __init__(self, taylor, ens_file, domain, init_cond, dtype=tf.float64, name = 'FPSolver', save_path=None, correction=None):
        super().__init__(name=name, dtype=dtype)
        self.ens_file = ens_file
        self.domain = Domain(domain, dtype=dtype)
        self.init_cond = init_cond
        hdf5 = tables.open_file(ens_file, 'r')
        self.final_available_time_id, self.time_step, self.ensemble_size, self.dim, self.num_runs = hdf5.root.config.read()[0]
        hdf5.close()
        self.current_time = -1
        self.normalizer = Normalizer(dtype=dtype)
        self.taylor = taylor(self.call, self.time_step)
        self.correction = correction
        self.folder = '{}/'.format(save_path) if save_path is not None else '' + '{}'.format(self.name)
        try:
            os.mkdir(self.folder)
        except:
            pass

        
    @ut.timer
    def summary(self):
        """
        Description:
            builds all layers and displays a summary of the network
        """
        x = tf.zeros((self.ensemble_size, 1), dtype=self.dtype)
        args = [x for _ in range(self.dim)]
        self(*args)
        self.taylor(*args)
        self.init_cond(*args[:])
        super().summary()

    def __summary__(self):
        super().summary()        

    @ut.timer
    def prepare(self):
        """
        Description:
            prepares for the next time step
        """
        # update the clock
        self.current_time += 1
        if self.current_time > 0:
            self.prev_ref_pts = copy.deepcopy(self.curr_ref_pts)
        hdf5_ens = tables.open_file(self.ens_file, 'r+')
        run = np.random.randint(self.num_runs)
        hdf5_ens = tables.open_file(self.ens_file, 'r+')
        for i in range(self.num_runs):
            if i > 0:
                pts = np.append(pts, getattr(getattr(hdf5_ens.root, 'run_{}'.format(i)).ensemble, 'time_{}'.format(self.current_time)).read(), axis=0)
            else:
                pts = getattr(getattr(hdf5_ens.root, 'run_{}'.format(i)).ensemble, 'time_{}'.format(self.current_time)).read()
        self.curr_ref_pts = tf.split(pts, self.dim, axis=1)
        self.domain_pts = self.domain.sample(self.ensemble_size)
        # generate data for training
        if self.current_time > 0:
            self.taylor.f = self.call
            self.prev_target_values = self.call(*self.prev_ref_pts)#copy.deepcopy(self.target_values)
            self.target_values = self.taylor(*self.domain_pts)
           
        else:
            self.target_values = self.init_cond(*self.domain_pts)#self.init_cond(*self.domain_pts)

        

        hdf5_ens.close()
        """
        #self.numerator = self.init_cond(*self.curr_intg_pts)
        if self.current_time % 5 == 4:
            hdf5_ens_helper = tables.open_file(self.ens_file[:-3] + '_helper.h5' , 'r')
            pts = getattr(hdf5_ens_helper.root.ensemble, 'time_' + str(self.current_time - 4)).read()
            self.curr_intg_pts = tf.split(pts, self.dim, axis=1)
            self.load_weights(self.current_time - 4)
            self.denom = self.call(*self.curr_intg_pts)
            self.load_weights(self.current_time - 1)
            hdf5_ens_helper.close()
        """

    @ut.timer
    def update(self, epochs=100, initial_rate=1e-3):
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
            #self.correction.f = self.call
            with tf.GradientTape() as tape:
                loss =  self.equality_loss() 
                print('epoch = {}, loss = {}'.format(epoch + 1, loss), end='\r')
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print('Invalid value encountered during computation of loss. Exiting training loop ...')
                    break
            grads = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
        

    @ut.timer
    def solve(self, final_time_id=None, initial_time_id=0, epochs_per_step=100, initial_rate=1e-3):
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
        """
        if final_time_id is None:
            final_time_id = self.final_available_time_id - 1
        else:
            final_time_id = min(self.final_available_time_id - 1, final_time_id)
        """
        plotter = pltr.NNPlotter(funcs=[self], space=self.domain.domain.numpy(), num_pts_per_dim=300)
        self.current_time = initial_time_id - 1
        x = tf.zeros((1, 1), dtype=self.dtype)
        y = tf.ones((1, 1), dtype=self.dtype)
        args = [x, y] #[x for _ in range(self.dim)]
        if initial_time_id != 0:
            self.load_weights(initial_time_id - 1)
        for i in range(initial_time_id, final_time_id + 1, 1):
            self.prepare()
            if i == 0:
                for _ in range(int(4000/epochs_per_step)):
                    self.update(epochs=epochs_per_step, initial_rate=initial_rate)
            else:
                self.update(epochs=epochs_per_step, initial_rate=initial_rate)
            print('learning done for time = {}, prob at attractor = {}'.format(self.current_time, self.call(*args)))
            self.save_weights()
            plotter.plot(self.folder + '/time_{}.png'.format(self.current_time), wireframe=True)
            #self.load_weights('random')

    def equality_loss(self):
        """
        Description:
            computes the Equality loss
        """
        integ_correction = 0. if self.current_time == 0 else 0.1 * tf.math.square(tf.reduce_mean(self.call(*self.prev_ref_pts)/self.prev_target_values) - 1.0)
        return tf.reduce_mean(tf.math.square(self.call(*self.domain_pts) - self.target_values)) +\
               integ_correction #tf.reduce_mean(tf.math.square(self.correction(*self.curr_ref_pts))) +- 0.001*tf.reduce_mean(self.call(*self.curr_ref_pts))\
        


    def correct(self):
        if self.current_time % 5 == 4:
            return (tf.reduce_mean(self.call(*self.curr_intg_pts)/self.denom) - 1.)**2 
        else:
            return 0.

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