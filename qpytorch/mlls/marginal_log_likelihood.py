#!/usr/bin/env python3

from ..models import GP, QEP
from ..module import Module


class MarginalLogLikelihood(Module):
    r"""
    These are modules to compute (or approximate/bound) the marginal log likelihood
    (MLL) of the GP (QEP) model when applied to data.  I.e., given a GP :math:`f \sim
    \mathcal{GP}(\mu, K)` or QEP :math:`f \sim \mathcal{QEP}(\mu, K)`, and 
    data :math:`\mathbf X, \mathbf y`, these modules compute/approximate

    .. math::

       \begin{equation*}
          \mathcal{L} = p_f(\mathbf y \! \mid \! \mathbf X)
          = \int p \left( \mathbf y \! \mid \! f(\mathbf X) \right) \: p(f(\mathbf X) \! \mid \! \mathbf X) \: d f
       \end{equation*}

    This is computed exactly when the GP (QEP) inference is computed exactly (e.g. regression w/ a Gaussian (Q-Exponential) likelihood).
    It is approximated/bounded for GP (QEP) models that use approximate inference.

    These models are typically used as the "loss" functions for GP (QEP) models (though note that the output of
    these functions must be negated for optimization).
    """

    def __init__(self, likelihood, model):
        super(MarginalLogLikelihood, self).__init__()
        if not isinstance(model, (GP, QEP)):
            raise RuntimeError(
                "All MarginalLogLikelihood objects must be given a GP (QEP) object as a model. If you are "
                "using a more complicated model involving a GP (QEP), pass the underlying GP (QEP) object as the "
                "model, not a full PyTorch module."
            )
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and `\mathbf y`

        :param ~gpytorch.distributions.MultivariateNormal or ~qpytorch.distributions.MultivariateQExponential 
            output: the outputs of the latent function (the :obj:`~gpytorch.models.GP` or :obj:`~qpytorch.models.QEP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param dict kwargs: Additional arguments to pass to the likelihood's forward function.
        """
        raise NotImplementedError
