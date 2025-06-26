#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import GridInterpolationKernel, ProductStructureKernel, RBFKernel, ScaleKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior

POWER = 1.0

# Simple training data: let's try to learn a sine function,
# but with KISS-QEP let's use 100 training examples.
n = 30
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_x = train_x
train_y = (torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])) * (2 * pi)

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_x = test_x
test_y = (torch.sin(test_x[:, 0]) + torch.cos(test_x[:, 1])) * (2 * pi)


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class QEPRegressionModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super(QEPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1, 1))
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = ProductStructureKernel(
            GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1), num_dims=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestKISSQEPMultiplicativeRegression(unittest.TestCase):
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

    def test_kissqep_qep_mean_abs_error(self):
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = QEPRegressionModel(train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        # Optimize the model
        qep_model.train()
        likelihood.train()

        optimizer = optim.Adam(qep_model.parameters(), lr=0.2)
        optimizer.n_iter = 0
        for _ in range(20):
            optimizer.zero_grad()
            output = qep_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.n_iter += 1

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

        # Test the model
        qep_model.eval()
        likelihood.eval()

        with qpytorch.settings.fast_pred_var():
            test_preds = likelihood(qep_model(test_x)).mean
        mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
        self.assertLess(mean_abs_error.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
