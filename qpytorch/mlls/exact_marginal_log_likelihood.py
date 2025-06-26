#!/usr/bin/env python3

from linear_operator.operators import MaskedLinearOperator

from .. import settings
from ..distributions import MultivariateNormal, MultivariateQExponential
from ..likelihoods import _GaussianLikelihoodBase, _QExponentialLikelihoodBase
from .marginal_log_likelihood import MarginalLogLikelihood


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian (Q-Exponential) process with a
    Gaussian (Q-Exponential) likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~qpytorch.likelihoods.GaussianLikelihood`
        (:obj:`~qpytorch.likelihoods.QExponentialLikelihood`) and a :obj:`~gpytorch.models.ExactGP` (:obj:`~qpytorch.models.ExactQEP`). 
        It also cannot be used in conjunction with stochastic optimization.

    :param ~qpytorch.likelihoods.GaussianLikelihood (~qpytorch.likelihoods.QExponentialLikelihood) likelihood: The Gaussian (Q-Exponential) likelihood for the model
    :param ~gpytorch.models.ExactGP (~qpytorch.models.ExactQEP) model: The exact GP (QEP) model

    Example:
        >>> # model is a qpytorch.models.ExactGP or qpytorch.models.ExactQEP
        >>> # likelihood is a qpytorch.likelihoods.Likelihood
        >>> mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(likelihood, (_GaussianLikelihoodBase, _QExponentialLikelihoodBase)):
            raise RuntimeError("Likelihood must be Gaussian or Q-Exponential for exact inference")
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        res_ndim = res.ndim
        for name, module, prior, closure, _ in self.model.named_priors():
            prior_term = prior.log_prob(closure(module))
            res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

        return res

    def forward(self, function_dist, target, *params, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal or ~qpytorch.distributions.MultivariateQExponential function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP` or :obj:`qpytorch.models.ExactQEP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, (MultivariateNormal, MultivariateQExponential)):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian or Q-Exponential random variables")

        # Determine output likelihood
        output = self.likelihood(function_dist, *params, **kwargs)

        # Remove NaN values if enabled
        if settings.observation_nan_policy.value() == "mask":
            observed = settings.observation_nan_policy._get_observed(target, output.event_shape)
            if isinstance(function_dist, MultivariateNormal):
                output = MultivariateNormal(
                    mean=output.mean[..., observed],
                    covariance_matrix=MaskedLinearOperator(
                        output.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                    ),
                )
            elif isinstance(function_dist, MultivariateQExponential):
                output = MultivariateQExponential(
                    mean=output.mean[..., observed],
                    covariance_matrix=MaskedLinearOperator(
                        output.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                    ),
                    power=output.power
                )
            target = target[..., observed]
        elif settings.observation_nan_policy.value() == "fill":
            raise ValueError("NaN observation policy 'fill' is not supported by ExactMarginalLogLikelihood!")

        # Get the log prob of the marginal distribution
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)
