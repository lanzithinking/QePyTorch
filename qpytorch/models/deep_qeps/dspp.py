#!/usr/bin/env python3

import torch
from linear_operator.operators import BlockDiagLinearOperator

from ... import settings
from ...distributions import MultitaskMultivariateQExponential
from ..approximate_qep import ApproximateQEP
from .deep_qep import DeepQEP, DeepQEPLayer


class DSPPLayer(DeepQEPLayer):
    """
    Represents a layer in a DSPP where inference is performed using the techniques defined in Jankowiak et al., 2020.

    From an end user perspective, the functionality and usage of a DSPPLayer is essentially identical to that of a
    DeepQEPLayer. It is therefore recommended that you review the documentation for DeepQEPLayer.

    :param ~gpytorch.variational.VariationalStrategy variational_strategy: Strategy for
        changing q(u) -> q(f) (see other VI docs)
    :param int input_dims: Dimensionality of input data expected by each QEP
    :param int output_dims: (default None) Number of QEPs in this layer, equivalent to
        output dimensionality. If set to `None`, then the output dimension will be squashed.
    :param int num_quad_sites: Number of quadrature sites to use. Also the number of Q-Exponentials in the mixture output
        by this layer.

    Again, refer to the documentation for DeepQEPLayer or our example notebooks for full details on what calling a
    DSPPLayer module does. The high level overview is that if a tensor `x` is `n x d` then

        >>> hidden_qep2(hidden_qep1(x))

    will return a `num_quad_sites` by `output_dims` set of Q-Exponentials, where for each output dim the first batch dim
    represents a weighted mixture of `num_quad_sites` Q-Exponentials with weights given by DSPP.quad_weights (see DSPP below)
    """

    def __init__(self, variational_strategy, input_dims, output_dims, num_quad_sites=3, quad_sites=None):
        super().__init__(variational_strategy, input_dims, output_dims)

        self.num_quad_sites = num_quad_sites

        # Pass in previous_layer.quad_sites if you want to share quad_sites across layers.
        if quad_sites is not None:
            self.quad_sites = quad_sites
        else:
            self.quad_sites = torch.nn.Parameter(torch.randn(num_quad_sites, input_dims))

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, MultitaskMultivariateQExponential):
            # This is for subsequent layers. We apply quadrature here
            # Mean, stdv are q x ... x n x t
            mus, sigmas = inputs.mean, inputs.variance.sqrt()
            qg = self.quad_sites.view([self.num_quad_sites] + [1] * (mus.dim() - 2) + [self.input_dims])
            sigmas = sigmas * qg
            inputs = mus + sigmas  # q^t x n x t
            deterministic_inputs = False
        else:
            deterministic_inputs = True

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

        # If this is the first layer (deterministic inputs), expand the output
        # This allows quadrature to be applied to future layers
        if deterministic_inputs:
            output = output.expand(torch.Size([self.num_quad_sites]) + output.batch_shape)

        if self.num_quad_sites > 0:
            if self.output_dims is not None and not isinstance(output, MultitaskMultivariateQExponential):
                mean = output.loc.transpose(-1, -2)
                covar = BlockDiagLinearOperator(output.lazy_covariance_matrix, block_dim=-3)
                output = MultitaskMultivariateQExponential(mean, covar, power=output.power, interleaved=False)
        else:
            output = output.loc.transpose(-1, -2)  # this layer provides noiseless kernel interpolation

        return output


class DSPP(DeepQEP):
    """
    A container module to build a DSPP
    This module should contain :obj:`~gpytorch.models.deep_qeps.DSPPLayer`
    modules, and can also contain other modules as well.

    This Module contains an additional set of parameters, `raw_quad_weights`, that represent the mixture weights for
    the output distribution.
    """

    def __init__(self, num_quad_sites):
        super().__init__()
        self.num_quad_sites = num_quad_sites
        self.register_parameter("raw_quad_weights", torch.nn.Parameter(torch.randn(self.num_quad_sites)))

    @property
    def quad_weights(self):
        qwd = self.raw_quad_weights
        return qwd - qwd.logsumexp(dim=-1)
