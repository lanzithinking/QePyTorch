#!/usr/bin/env python3

import os
import random
import unittest
from math import exp

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import RBFKernel, ScaleKernel
from qpytorch.likelihoods import BernoulliLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior

POWER = 1.0

n = 4
train_x = torch.zeros(pow(n, 2), 2)
train_y = torch.zeros(pow(n, 2))
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
        train_y[i * n + j] = pow(-1, int(i / 2) + int(j / 2))
train_x = train_x
train_y = train_y.add(1).div(2)


class QEPClassificationModel(qpytorch.models.ApproximateQEP):
    def __init__(self, grid_size=6, grid_bounds=[(-0.33, 1.33), (-0.33, 1.33)]):
        self.power = torch.tensor(POWER)
        variational_distribution = qpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=int(pow(grid_size, len(grid_bounds))), power=self.power
        )
        variational_strategy = qpytorch.variational.GridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=grid_bounds, variational_distribution=variational_distribution
        )
        super(QEPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=2, lengthscale_prior=SmoothedBoxPrior(exp(-2.5), exp(3), sigma=0.1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateQExponential(mean_x, covar_x, power=self.power)
        return latent_pred


class TestKISSQEPKroneckerProductClassification(unittest.TestCase):
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

    def test_kissqep_classification_error(self):
        model = QEPClassificationModel()
        likelihood = BernoulliLikelihood()
        mll = qpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        with qpytorch.settings.max_preconditioner_size(5):
            optimizer = optim.Adam(model.parameters(), lr=0.15)
            optimizer.n_iter = 0
            for _ in range(100):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Set back to eval mode
            model.eval()
            likelihood.eval()

            test_preds = model(train_x).mean.ge(0.5).float()
            mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
            self.assertLess(mean_abs_error.squeeze().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
