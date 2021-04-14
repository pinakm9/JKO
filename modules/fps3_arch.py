import tensorflow as tf
import fps3 as fp 


class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float64):
        super().__init__(name='LSTMForgetBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c_ = tf.keras.activations.tanh(self.W_c(x) + self.U_c(h))
        c = f*c + i*c_
        return o*tf.keras.activations.tanh(c), c


class LSTMPeepholeBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float64):
        super().__init__(name='LSTMPeepholeBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c = f*c + i * tf.keras.activations.tanh(self.W_c(x))
        return o*c, c

class DGMBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float64):
        super().__init__(name='DGMBlock', dtype=dtype)
        self.W_z = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_z = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_h = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_h = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, S):
        Z = tf.keras.activations.sigmoid(self.W_z(S) + self.U_z(x))
        G = tf.keras.activations.sigmoid(self.W_g(S) + self.U_g(x))
        R = tf.keras.activations.sigmoid(self.W_r(S) + self.U_r(x))
        H = tf.keras.activations.tanh(self.W_h(S*R) + self.U_h(x))
        return (1.0 - G)*H + Z*S

class FPForget(fp.FPSolver):
    """
    Description: 
        LSTM Forget architecture for the FPSolver, inherits from FPSolver
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
        ------ parent args ------
        diff_ops: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
    """
    def __init__(self, num_nodes, num_layers, taylor, ens_file, domain, init_cond, dtype=tf.float64,\
                 name = 'FPForget', save_path=None):
        super().__init__(taylor, ens_file, domain, init_cond, dtype, name, save_path)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return self.normalizer(y)


class FPPeephole(fp.FPSolver):
    """
    Description: 
        LSTM Peephole architecture for the FPSolver, inherits from FPSolver
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
        ------ parent args ------
        diff_ops: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
    """
    def __init__(self, num_nodes, num_layers, taylor, ens_file, domain, init_cond, dtype=tf.float64,\
                 name = 'FPPeephole', save_path=None):
        super().__init__(taylor, ens_file, domain, init_cond, dtype, name, save_path)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMPeepholeBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return self.normalizer(y)

class FPDGM(fp.FPSolver):
    """
    Description: 
        DGM architecture for the FPSolver, inherits from FPSolver
    Args:
        num_nodes: number of nodes in each DGM layer
        num_layers: number of DGM layers
        ------ parent args ------
        diff_ops: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
    """
    def __init__(self, num_nodes, num_layers, taylor, ens_file, domain, init_cond, dtype=tf.float64,\
                 name = 'FPDGM', save_path=None):
        super().__init__(taylor, ens_file, domain, init_cond, dtype, name, save_path)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.initial_dense = tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh, dtype=dtype)
        self.lstm_layers = [DGMBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
         

    def call(self, *args):
        x = tf.concat(args, axis=1)
        S = self.initial_dense(x)
        for i in range(self.num_layers):
            S = self.lstm_layers[i](x, S)
        y = self.final_dense(S)
        return self.normalizer(y)


class FPVanilla(fp.FPSolver):
    """
    Description: 
        Vanilla architecture for the FPSolver, inherits from FPSolver
    Args:
        num_nodes: number of nodes in each Vanilla layer
        num_layers: number of Vanilla layers
        ------ parent args ------
        diff_ops: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
    """
    def __init__(self, num_nodes, num_layers, taylor, ens_file, domain, init_cond, dtype=tf.float64,\
                 name = 'FPVanilla', save_path=None):
        super().__init__(taylor, ens_file, domain, init_cond, dtype, name, save_path)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.initial_dense = tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh, dtype=dtype)
        self.middle_layers = [tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
         

    def call(self, *args):
        x = self.initial_dense(tf.concat(args, axis=1))
        for i in range(self.num_layers):
            x = self.middle_layers[i](x)
        x = self.final_dense(x)
        return self.normalizer(x)