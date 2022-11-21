
import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Gamma, Normal, MultivariateNormal



class NormalInverseGamma(Distribution):
    """
    A distribution `NIG(alpha, beta, delta, mu)` which is a normal variance-mean mixture with the inverse Gaussian distribution for mixing.
    """

    def __init__(self, gamma: Tensor, v: Tensor, alpha: Tensor, beta: Tensor):
        self.gamma = gamma
        self.v = v
        self.alpha = alpha
        self.beta = beta

        raise NotImplementedError("TODO: implement a full torch.Distribution child class")