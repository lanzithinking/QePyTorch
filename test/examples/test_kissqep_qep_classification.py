#!/usr/bin/env python3

import os
import random
import unittest
from math import exp, pi

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import RBFKernel, ScaleKernel
from qpytorch.likelihoods import BernoulliLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior

POWER = 1.0

train_x = torch.linspace(0, 1, 10)
train_y = torch.sign(torch.cos(train_x * (16 * pi))).add(1).div(2)


class QEPClassificationModel(qpytorch.models.ApproximateQEP):
    def __init__(self, grid_size=32, grid_bounds=[(0, 1)]):
        self.power = torch.tensor(POWER)
        variational_distribution = qpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=int(pow(grid_size, len(grid_bounds))), power=self.power,
        )
        variational_strategy = qpytorch.variational.GridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=grid_bounds, variational_distribution=variational_distribution
        )
        super(QEPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-5, 5))
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)),
            outputscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateQExponential(mean_x, covar_x, power=self.power)
        return latent_pred


class TestKISSQEPClassification(unittest.TestCase):
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

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.n_iter = 0
        for _ in range(200):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        # Set back to eval mode
        model.eval()
        likelihood.eval()
        test_preds = likelihood(model(train_x)).mean.ge(0.5).float()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        self.assertLess(mean_abs_error.squeeze().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
