#!/usr/bin/env python3

import os
import random
import unittest
import warnings
from math import pi

import linear_operator
import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import RFFKernel, ScaleKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior
from qpytorch.utils.warnings import NumericalWarning

POWER = 1.0

# Simple training data: let's try to learn a sine function,
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
        self.covar_module = ScaleKernel(RFFKernel(num_samples=10))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestRFFRegression(unittest.TestCase):
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

    def test_rff_mean_abs_error(self):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        train_x, train_y, test_x, test_y = make_data()
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = QEPRegressionModel(train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

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

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)
