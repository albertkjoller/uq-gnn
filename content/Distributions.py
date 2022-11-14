
import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Gamma, Normal

class NormalInverseGamma(Distribution):
    """
    A distribution `NIG(alpha, beta, delta, mu)` which is a normal variance-mean mixture with the inverse Gaussian distribution for mixing.
    """

    def __init__(self, gamma: Tensor, v: Tensor, alpha: Tensor, beta: Tensor):
        #assert alpha.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        # TODO: implement assertion statement for checking dimensionality
        self.gamma = gamma
        self.v = v
        self.alpha = alpha
        self.beta = beta

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()
        # raise NotImplementedError # <- your content

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""

        # nll = #TODO: check appendix of the paper and at this
        # https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/losses/continuous.py
        pass

# TODO:
# 1) easy way -->        just implement the loss following the formulas in the paper
# 2) better way -->   implement a full NIG probability distribution from which sampling, etc. is possible
        # 2.1) What are we going to sample? Parameters for the likelihood?



        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)
