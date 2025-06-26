#!/usr/bin/env python3

import unittest
from math import pi

import torch

import qpytorch
from qpytorch.distributions import MultitaskMultivariateQExponential
from qpytorch.likelihoods import MultitaskQExponentialLikelihood
from qpytorch.means import ConstantMean, MultitaskMean
from qpytorch.test import BaseTestCase

POWER = 2.0

# Simple training data: let's try to learn a sine function
train_x = torch.linspace(0, 1, 100)

# y1 function is sin(2*pi*x) with noise N(0, 0.04)
train_y1 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.1
# y2 function is cos(2*pi*x) with noise N(0, 0.04)
train_y2 = torch.cos(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.1

# Create a train_y which interleaves the two
train_y = torch.stack([train_y1, train_y2], -1)


class MultitaskQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskQEPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.covar_module = qpytorch.kernels.MultitaskKernel(
            qpytorch.kernels.InducingPointKernel(
                qpytorch.kernels.RBFKernel(), inducing_points=torch.randn(50, 1), likelihood=likelihood
            ),
            num_tasks=2,
            rank=2,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestSimpleQEPRegression(BaseTestCase, unittest.TestCase):
    seed = 0

    def test_multitask_qep_mean_abs_error(self):
        likelihood = MultitaskQExponentialLikelihood(num_tasks=2, power=torch.tensor(POWER))
        model = MultitaskQEPModel(train_x, train_y, likelihood)
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes QExponentialLikelihood parameters

        # "Loss" for QEPs - the marginal log likelihood
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        n_iter = 50
        for _ in range(n_iter):
            # Zero prev backpropped gradients
            optimizer.zero_grad()
            # Make predictions from training data
            # Again, note feeding duplicated x_data and indices indicating which task
            output = model(train_x)
            # TODO: Fix this view call!!
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        likelihood.eval()
        test_x = torch.linspace(0, 1, 51)
        test_y1 = torch.sin(test_x * (2 * pi))
        test_y2 = torch.cos(test_x * (2 * pi))
        test_preds = likelihood(model(test_x)).mean
        mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds[:, 0]))
        mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds[:, 1]))

        self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
