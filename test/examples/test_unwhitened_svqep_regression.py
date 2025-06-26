#!/usr/bin/env python3

import math
import unittest

import torch
from torch import optim

import qpytorch
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.models import ApproximateQEP
from qpytorch.test import BaseTestCase

POWER = 1.0

def train_data(cuda=False):
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * math.pi))
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class SVQEPRegressionModel(ApproximateQEP):
    def __init__(self, inducing_points, distribution_cls):
        self.power = torch.tensor(POWER)
        variational_distribution = distribution_cls(inducing_points.size(-1), power=self.power)
        variational_strategy = qpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True, jitter_val=1e-4
        )
        super(SVQEPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = qpytorch.means.ConstantMean()
        self.covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power)
        return latent_pred


class TestSVQEPRegression(BaseTestCase, unittest.TestCase):
    seed = 0

    def test_regression_error(
        self,
        cuda=False,
        mll_cls=qpytorch.mlls.VariationalELBO,
        distribution_cls=qpytorch.variational.CholeskyVariationalDistribution,
    ):
        train_x, train_y = train_data(cuda=cuda)
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        model = SVQEPRegressionModel(torch.linspace(0, 1, 25), distribution_cls)
        mll = mll_cls(likelihood, model, num_data=len(train_y))
        if cuda:
            likelihood = likelihood.cuda()
            model = model.cuda()
            mll = mll.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        for _ in range(200):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for param in likelihood.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        # Set back to eval mode
        model.eval()
        likelihood.eval()
        test_preds = likelihood(model(train_x)).mean.squeeze()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        self.assertLess(mean_abs_error.item(), 0.014)

        if distribution_cls is qpytorch.variational.CholeskyVariationalDistribution:
            # finally test fantasization
            # we only will check that tossing the entire training set into the model will reduce the mae
            model.likelihood = likelihood
            fant_model = model.get_fantasy_model(train_x, train_y)
            fant_preds = fant_model.likelihood(fant_model(train_x)).mean.squeeze()
            updated_abs_error = torch.mean(torch.abs(train_y - fant_preds) / 2)
            # TODO: figure out why this error is worse than before
            self.assertLess(updated_abs_error.item(), 0.15)


if __name__ == "__main__":
    unittest.main()
