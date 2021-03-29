import tensorflow as tf
import jko_solver as jko 


class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float32):
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
        c = tf.keras.activations.tanh(f*c + i*c_)
        return o*c, c


class JKOLSTM(jko.JKOSolver):
    """
    Description: 
        LSTM architecture for the JKOSolver, inherits from JKOSolver
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
        ------ parent args ------
        psi: potential that is a function of space
        beta: scalar whose inverse denotes the "intensity" of noise
        ens_file: path to ensemble evolution file
        cost_file: path to cost file associated with the ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the JKOSolver network
    """
    def __init__(self, num_nodes, num_layers, psi, beta, ens_file, cost_file, sinkhorn_epsilon=0.01, sinkhorn_iters=100, name = 'JKOLSTM'):
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        dtype = tf.float32
        super().__init__(psi, beta, ens_file, cost_file, sinkhorn_epsilon, sinkhorn_iters, dtype, name)
        self.lstm_layers = [LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, x):
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            h = self.batch_norm(h)
            c = self.batch_norm(c)
        return self.final_dense(h)

