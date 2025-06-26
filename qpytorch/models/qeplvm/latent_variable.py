#!/usr/bin/env python3

import torch

from ...module import Module


class LatentVariable(Module):
    """
    This super class is used to describe the type of inference
    used for the latent variable :math:`\\mathbf X` in QEPLVM models.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim

    def forward(self, x):
        raise NotImplementedError


class PointLatentVariable(LatentVariable):
    """
    This class is used for QEPLVM models to recover a MLE estimate of
    the latent variable :math:`\\mathbf X`.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    """

    def __init__(self, n, latent_dim, X_init):
        super().__init__(n, latent_dim)
        self.register_parameter("X", X_init)

    def forward(self):
        return self.X


class MAPLatentVariable(LatentVariable):
    """
    This class is used for QEPLVM models to recover a MAP estimate of
    the latent variable :math:`\\mathbf X`, based on some supplied prior.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    :param ~gpytorch.priors.Prior prior_x: prior for :math:`\\mathbf X`
    """

    def __init__(self, n, latent_dim, X_init, prior_x):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.register_parameter("X", X_init)
        self.register_prior("prior_x", prior_x, "X")

    def forward(self):
        return self.X


class VariationalLatentVariable(LatentVariable):
    """
    This class is used for QEPLVM models to recover a variational approximation of
    the latent variable :math:`\\mathbf X`. The variational approximation will be
    an isotropic Q-Exponential distribution.

    :param int n: Size of the latent space.
    :param int data_dim: Dimensionality of the :math:`\\mathbf Y` values.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    :param ~gpytorch.priors.Prior prior_x: prior for :math:`\\mathbf X`
    """

    def __init__(self, n, data_dim, latent_dim, X_init, prior_x, **kwargs):
        super().__init__(n, latent_dim)

        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianQEPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")
        
        self.power = kwargs.pop('power', getattr(self.prior_x, 'power', torch.tensor(2.0)))

    def forward(self):
        from ...distributions import QExponential
        from ...mlls import KLQExponentialAddedLossTerm

        # Variational distribution over the latent variable q(x)
        q_x = QExponential(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma), power=self.power)
        x_kl = KLQExponentialAddedLossTerm(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term("x_kl", x_kl)  # Update the KL term
        return q_x.rsample()
