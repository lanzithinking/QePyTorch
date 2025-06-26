#!/usr/bin/env python3

import math
import unittest
from unittest.mock import MagicMock, patch

import linear_operator
import torch
from torch import optim

import qpytorch
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.models import ApproximateQEP
from qpytorch.test import BaseTestCase

POWER = 1.99

def train_data():
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * math.pi))
    return train_x, train_y


class SVQEPRegressionModel(ApproximateQEP):
    def __init__(self, inducing_points, base_inducing_points):
        self.power = torch.tensor(POWER)
        base_variational_distribution = qpytorch.variational.CholeskyVariationalDistribution(
            base_inducing_points.size(-1), power=self.power
        )
        variational_distribution = qpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-1))
        variational_strategy = qpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            qpytorch.variational.VariationalStrategy(
                self,
                base_inducing_points,
                base_variational_distribution,
                learn_inducing_locations=True,
                jitter_val=1e-4,
            ),
            inducing_points,
            variational_distribution,
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
        mll_cls=qpytorch.mlls.VariationalELBO,
        distribution_cls=qpytorch.variational.CholeskyVariationalDistribution,
    ):
        train_x, train_y = train_data()
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        model = SVQEPRegressionModel(torch.linspace(0, 1, 128), torch.linspace(0, 1, 16))
        mll = mll_cls(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        _cg_mock = patch("linear_operator.utils.linear_cg", new=_wrapped_cg)
        with _cg_mock as cg_mock:
            for _ in range(75):
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
            self.assertLess(mean_abs_error.item(), 1e-1)

            self.assertFalse(cg_mock.called)

    def test_predictive_ll_regression_error(self):
        return self.test_regression_error(
            mll_cls=qpytorch.mlls.PredictiveLogLikelihood,
            distribution_cls=qpytorch.variational.MeanFieldVariationalDistribution,
        )


if __name__ == "__main__":
    unittest.main()
