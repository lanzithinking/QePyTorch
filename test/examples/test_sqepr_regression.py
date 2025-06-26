#!/usr/bin/env python3

import os
import random
import unittest
import warnings
from math import exp, pi
from unittest.mock import MagicMock, patch

import linear_operator
import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior

from qpytorch.test import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device
from qpytorch.utils.warnings import NumericalWarning

POWER = 2.0

# Simple training data: let's try to learn a sine function,
# but with SQEPR
# let's use 100 training examples.
def make_data(cuda=False):
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * pi))
    train_y.add_(torch.randn_like(train_y), alpha=1e-2)
    test_x = torch.rand(51)
    test_y = torch.sin(test_x * (2 * pi))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class QEPRegressionModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super(QEPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)))
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=torch.linspace(0, 1, 32), likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestSQEPRRegression(unittest.TestCase, BaseTestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_sgpr_mean_abs_error(self, cuda=False):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        train_x, train_y, test_x, test_y = make_data(cuda=cuda)
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = QEPRegressionModel(train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        if cuda:
            qep_model = qep_model.cuda()
            likelihood = likelihood.cuda()

        # Mock cholesky
        _wrapped_cholesky = MagicMock(wraps=torch.linalg.cholesky_ex)
        with patch("torch.linalg.cholesky_ex", new=_wrapped_cholesky) as cholesky_mock:

            # Optimize the model
            qep_model.train()
            likelihood.train()

            optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
            for _ in range(30):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                # Check that we have the right LinearOperator type
                kernel = likelihood(qep_model(train_x)).lazy_covariance_matrix.evaluate_kernel()
                self.assertIsInstance(kernel, linear_operator.operators.LowRankRootAddedDiagLinearOperator)

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            qep_model.eval()
            likelihood.eval()

            test_preds = likelihood(qep_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
            cholesky_mock.assert_called()  # We SHOULD call Cholesky...
            for chol_arg in cholesky_mock.call_args_list:
                first_arg = chol_arg[0][0]
                self.assertTrue(torch.is_tensor(first_arg))
                self.assertTrue(first_arg.size(-1) == qep_model.covar_module.inducing_points.size(-2))

        self.assertLess(mean_abs_error.squeeze().item(), 0.1)

        # Test variances
        test_vars = likelihood(qep_model(test_x)).variance
        self.assertAllClose(test_vars, likelihood(qep_model(test_x)).covariance_matrix.diagonal(dim1=-1, dim2=-2))
        self.assertGreater(test_vars.min().item() + 0.1, likelihood.noise.item())
        self.assertLess(
            test_vars.max().item() - 0.05,
            likelihood.noise.item() + qep_model.covar_module.base_kernel.outputscale.item(),
        )

        # Test on training data
        test_outputs = likelihood(qep_model(train_x))
        self.assertLess((test_outputs.mean - train_y).max().item(), 0.1)
        self.assertLess(test_outputs.variance.max().item(), likelihood.noise.item() * 2)

    def test_sgpr_mean_abs_error_cuda(self):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        if not torch.cuda.is_available():
            return

        with least_used_cuda_device():
            self.test_sgpr_mean_abs_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
