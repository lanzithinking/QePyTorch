#!/usr/bin/env python3

import unittest
from math import pi

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import ScaleKernel
from gpytorch.kernels.keops import RBFKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.test import BaseTestCase

POWER = 1.0

# Simple training data: let's try to learn a sine function
train_x = torch.randn(1000, 2)
train_y = torch.sin(train_x[..., 0] * (2 * pi) + train_x[..., 1])
train_y = train_y + torch.randn_like(train_y).mul(0.001)

test_x = torch.randn(50, 2)
test_y = torch.sin(test_x[..., 0] * (2 * pi) + test_x[..., 1])


class KeOpsQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestKeOpsQEPRegression(BaseTestCase, unittest.TestCase):
    seed = 4

    def test_keops_qep_mean_abs_error(self):
        try:
            import pykeops  # noqa
        except ImportError:
            return

        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = KeOpsQEPModel(train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        # Optimize the model
        qep_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(qep_model.parameters()), lr=0.01)
        optimizer.n_iter = 0

        with qpytorch.settings.max_cholesky_size(0):  # Ensure that we're using KeOps
            for i in range(300):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                if i == 0:
                    for param in qep_model.parameters():
                        self.assertTrue(param.grad is not None)

            # Test the model
            with torch.no_grad():
                qep_model.eval()
                likelihood.eval()
                test_preds = likelihood(qep_model(test_x)).mean
                mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
