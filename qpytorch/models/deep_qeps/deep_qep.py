#!/usr/bin/env python3

import warnings

import torch
from linear_operator.operators import BlockDiagLinearOperator

from ... import settings
from ...distributions import QExponential, MultitaskMultivariateQExponential
from ...likelihoods import Likelihood
from ..approximate_qep import ApproximateQEP
from ..qep import QEP


class _DeepQEPVariationalStrategy(object):
    def __init__(self, model):
        self.model = model

    @property
    def sub_variational_strategies(self):
        if not hasattr(self, "_sub_variational_strategies_memo"):
            self._sub_variational_strategies_memo = [
                module.variational_strategy for module in self.model.modules() if isinstance(module, ApproximateQEP)
            ]
        return self._sub_variational_strategies_memo

    def kl_divergence(self):
        return sum(strategy.kl_divergence().sum() for strategy in self.sub_variational_strategies)


class DeepQEPLayer(ApproximateQEP):
    """
    Represents a layer in a deep QEP where inference is performed via the doubly stochastic method of
    Salimbeni et al., 2017. Upon calling, instead of returning a variational distribution q(f), returns samples
    from the variational distribution.

    See the documentation for __call__ below for more details below. Note that the behavior of __call__
    will change to be much more elegant with multiple batch dimensions; however, the interface doesn't really
    change.

    :param ~gpytorch.variational.VariationalStrategy variational_strategy: Strategy for
        changing q(u) -> q(f) (see other VI docs)
    :param int input_dims`: Dimensionality of input data expected by each QEP
    :param int output_dims: (default None) Number of QEPs in this layer, equivalent to
        output dimensionality. If set to `None`, then the output dimension will be squashed.

    Forward data through this hidden QEP layer. The output is a MultitaskMultivariateQExponential distribution
    (or MultivariateQExponential distribution is output_dims=None).

    If the input is >=2 dimensional Tensor (e.g. `n x d`), we pass the input through each hidden QEP,
    resulting in a `n x h` multitask Q-Exponential distribution (where all of the `h` tasks represent an
    output dimension and are independent from one another).  We then draw `s` samples from these Q-Exponentials,
    resulting in a `s x n x h` MultitaskMultivariateQExponential distribution.

    If the input is a >=3 dimensional Tensor, and the `are_samples=True` kwarg is set, then we assume that
    the outermost batch dimension is a samples dimension. The output will have the same number of samples.
    For example, a `s x b x n x d` input will result in a `s x b x n x h` MultitaskMultivariateQExponential distribution.

    The goal of these last two points is that if you have a tensor `x` that is `n x d`, then

        >>> hidden_qep2(hidden_qep(x))

    will just work, and return a tensor of size `s x n x h2`, where `h2` is the output dimensionality of
    hidden_qep2. In this way, hidden QEP layers are easily composable.
    """

    def __init__(self, variational_strategy, input_dims, output_dims):
        super(DeepQEPLayer, self).__init__(variational_strategy)
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, inputs, are_samples=False, **kwargs):
        deterministic_inputs = not are_samples
        if isinstance(inputs, MultitaskMultivariateQExponential):
            inputs = QExponential(loc=inputs.mean, scale=inputs.variance.sqrt(), power=inputs.power).rsample(rescale=kwargs.pop('rescale', False))
            deterministic_inputs = False

        if settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateQExponential or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.size(-1) != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        # Repeat the input for all possible outputs
        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        # Now run samples through the QEP
        output = ApproximateQEP.__call__(self, inputs, **kwargs)
        if self.output_dims is not None:
            mean = output.loc.transpose(-1, -2)
            covar = BlockDiagLinearOperator(output.lazy_covariance_matrix, block_dim=-3)
            output = MultitaskMultivariateQExponential(mean, covar, power=output.power, interleaved=False)

        # Maybe expand inputs?
        if deterministic_inputs:
            output = output.expand(torch.Size([settings.num_likelihood_samples.value()]) + output.batch_shape)

        return output


class DeepQEP(QEP):
    """
    A container module to build a DeepQEP.
    This module should contain :obj:`~gpytorch.models.deep.DeepQEPLayer`
    modules, and can also contain other modules as well.
    """

    def __init__(self):
        super().__init__()
        self.variational_strategy = _DeepQEPVariationalStrategy(self)

    def forward(self, x):
        raise NotImplementedError


class DeepLikelihood(Likelihood):
    """
    A wrapper to make a GPyTorch likelihood compatible with Deep QEPs

    Example:
        >>> deep_qexponential_likelihood = gpytorch.likelihoods.DeepLikelihood(gpytorch.likelihood.QExponentialLikelihood)
    """

    def __init__(self, base_likelihood):
        super().__init__()
        warnings.warn(
            "DeepLikelihood is now deprecated. Use a standard likelihood in conjunction with a "
            "gpytorch.mlls.DeepApproximateMLL. See the DeepQEP example in our documentation.",
            DeprecationWarning,
        )
        self.base_likelihood = base_likelihood

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        return self.base_likelihood.expected_log_prob(observations, function_dist, *params, **kwargs).mean(dim=0)

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        return self.base_likelihood.log_marginal(observations, function_dist, *params, **kwargs).mean(dim=0)

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.base_likelihood.__call__(*args, **kwargs)
