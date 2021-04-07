import numpy as np
import tables
import tensorflow as tf

def compute_cost_matrix(ensemble_1, ensemble_2, p=2):
    """
    Description:
        calculates the cost matrix for Wasserstein distance calculation
    Args:
        ensemble_1: first ensemble for Wasserstein distance calculation
        ensemble_2: second ensemble for Wasserstein distance calculation  
        p: p-norm to be used, default=2
    Returns:
        the cost matrix 
    """
    cost = np.zeros((len(ensemble_1), len(ensemble_2)))
    for i, x in enumerate(ensemble_1):
        for j, y in enumerate(ensemble_2):
            cost[i][j] = (np.abs(x-y)**p).sum()
    return cost

def compute_cost_evolution(ens_file, save_path, p=2):
    """
    Description:
        calculates cost matrices and saves them for an ensemble evolution
    Args:
        ens_file: path to .h5 file containing ensemble evolution
        save_path: path to .h5 file where the cost matricesa are to be saved
        p: p-norm to be used, default=2
    """
    hdf5_ens = tables.open_file(ens_file, 'r')
    hdf5_cost = tables.open_file(save_path, 'w')
    num_steps = hdf5_ens.root.config.read()[0][0]
    for time_id in range(num_steps):
        ensemble_1 = getattr(hdf5_ens.root.ensemble, 'time_' + str(time_id)).read()
        ensemble_2 = getattr(hdf5_ens.root.ensemble, 'time_' + str(time_id + 1)).read()
        cost = compute_cost_matrix(ensemble_1, ensemble_2, p)
        hdf5_cost.create_array(hdf5_cost.root, 'time_' + str(time_id), cost)
    hdf5_ens.close()
    hdf5_cost.close()

def compute_cost_evolution_fp(ens_file, save_path, p=2):
    """
    Description:
        calculates cost matrices and saves them for an ensemble evolution
    Args:
        ens_file: path to .h5 file containing ensemble evolution
        save_path: path to .h5 file where the cost matricesa are to be saved
        p: p-norm to be used, default=2
    """
    hdf5_ens = tables.open_file(ens_file, 'r')
    hdf5_cost = tables.open_file(save_path, 'w')
    num_steps = hdf5_ens.root.config.read()[0][0]
    for time_id in range(num_steps + 1):
        ensemble = getattr(hdf5_ens.root.ensemble, 'time_' + str(time_id)).read()
        cost = compute_cost_matrix(ensemble, ensemble, p)
        hdf5_cost.create_array(hdf5_cost.root, 'time_' + str(time_id), cost)
    hdf5_ens.close()
    hdf5_cost.close()

def sinkhorn_loss(ensemble_1, ensemble_2, weights_1, weights_2, cost_matrix, epsilon=0.01, num_iters=200):
    """
    Description:
        Given two weighted ensembles ensemble_1 and ensemble_2, 
        outputs an approximation of the OT cost with regularization parameter epsilon
        num_iters is the max. number of steps in sinkhorn loop
    
    Args:
        ensemble_1:  first ensemble as a tensor
        ensemble_2: second ensemble as a tensor
        weights_1: weights for the first ensemble
        weights_2: weights for the second ensemble
        cost_matrix: cost matrix for Sinkhorn iterations
        epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
        num_iters:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
    
    Returns:
        The optimal cost or the (Wasserstein distance) ** p
    """
    
    # Elementary operations
    def M(u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost_matrix + tf.expand_dims(u, 1) + tf.expand_dims(v, 0))/epsilon
    
    def lse(A):
        return tf.reduce_logsumexp(A, axis=1, keepdims=True)
    
    # The Sinkhorn loop
    u, v = tf.zeros_like(weights_1), tf.zeros_like(weights_2)
    for _ in range(num_iters):
        u = epsilon * (tf.math.log(weights_1) - tf.squeeze(lse(M(u, v)) )  ) + u
        v = epsilon * (tf.math.log(weights_2) - tf.squeeze( lse(tf.transpose(M(u, v))) ) ) + v
    
    cost = tf.reduce_sum(tf.exp(M(u, v)) * cost_matrix)
    return cost