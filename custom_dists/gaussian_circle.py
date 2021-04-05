import numpy as np 
import scipy.stats as ss

class GaussianCircle:
    """
    Description:
        creates a multimodal distribution aranged on a circle uniformly using iid Gaussians
    Args:
        mean: mean for each Gaussian distribution
        cov: covarinace matrix for each Gaussian distribution
        weights: a 1d array
    """
    def __init__(self, cov, weights):
        self.cov = cov 
        self.weights = weights / weights.sum()
        self.num_modes = len(weights)
        self.dim = cov.shape[0]
        self.means = np.zeros((self.num_modes, self.dim))
        angle = 2.0 * np.pi / self.num_modes
        for i in range(self.num_modes):
            self.means[i, :2] = np.cos(i * angle), np.sin(i * angle)

    def sample(self, size):
        """
        Description:
            samples from the multimodal distribtion
        Args:
            size: number of samples to be generated
        Returns:
             the generated samples
        """
        samples = np.zeros((size, self.dim))
        idx = np.random.choice(self.num_modes, size=size, replace=True, p=self.weights)
        for i in range(size):
            samples[i, :] = np.random.multivariate_normal(mean=self.means[idx[i]], cov=self.cov, size=1)
        return samples

    def pdf(self, x):
        """
        Description:
            computes probability for given samples
        Args:
            x: samples at which probability is to be computed
        Returns:
             the computed probabilities
        """
        probs =  np.zeros(x.shape[0])
        for i in range(self.num_modes):
            probs += self.weights[i] * ss.multivariate_normal.pdf(x, mean=self.means[i], cov=self.cov)
        return probs