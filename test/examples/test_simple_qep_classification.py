#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import torch
from torch import optim

import qpytorch
from qpytorch.likelihoods import BernoulliLikelihood
from qpytorch.models import ApproximateQEP
from qpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy

POWER = 1.9

def train_data():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sign(torch.cos(train_x * (4 * pi))).add(1).div(2)
    return train_x, train_y


class QEPClassificationModel(ApproximateQEP):
    def __init__(self, train_x):
        self.power = torch.tensor(POWER)
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0), power=self.power)
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(QEPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = qpytorch.means.ConstantMean()
        self.covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power)
        return latent_pred


class TestSimpleQEPClassification(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_classification_error(self):
        train_x, train_y = train_data()
        likelihood = BernoulliLikelihood()
        model = QEPClassificationModel(train_x)
        mll = qpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(75):
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
        test_preds = likelihood(model(train_x)).mean.round()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        self.assertLess(mean_abs_error.item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
