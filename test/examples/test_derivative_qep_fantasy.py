#!/usr/bin/env python3

import unittest
from math import pi

import torch

import qpytorch
from qpytorch.distributions import MultitaskMultivariateQExponential
from qpytorch.kernels import ScaleKernel, RBFKernelGrad
from qpytorch.likelihoods import MultitaskQExponentialLikelihood
from qpytorch.means import ConstantMeanGrad
from qpytorch.test import BaseTestCase

POWER = 1.0

# Simple training data
num_train_samples = 15
num_fantasies = 10
dim = 1
train_X = torch.linspace(0, 1, num_train_samples).reshape(-1, 1)
train_Y = torch.hstack([
    torch.sin(train_X * (2 * pi)).reshape(-1, 1),
    (2 * pi) * torch.cos(train_X * (2 * pi)).reshape(-1, 1),
])


class QEPWithDerivatives(qpytorch.models.ExactQEP):
    def __init__(self, train_X, train_Y):
        likelihood = MultitaskQExponentialLikelihood(num_tasks=1 + dim, power=torch.tensor(POWER))
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMeanGrad()
        self.base_kernel = RBFKernelGrad()
        self.covar_module = ScaleKernel(self.base_kernel)
        self._num_outputs = 1 + dim

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestDerivativeQEPFutures(BaseTestCase, unittest.TestCase):

    # Inspired by test_lanczos_fantasy_model
    def test_derivative_qep_futures(self):
        model = QEPWithDerivatives(train_X, train_Y)
        mll = qpytorch.mlls.sum_marginal_log_likelihood.ExactMarginalLogLikelihood(model.likelihood, model)

        mll.train()
        mll.eval()

        # get a posterior to fill in caches
        model(torch.randn(num_train_samples).reshape(-1, 1))
        
        # hack: redirect the prediction_strategy
        model.prediction_strategy.__class__ = qpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy

        new_x = torch.randn((1, 1, dim))
        new_y = torch.randn((num_fantasies, 1, 1, 1 + dim))

        # just check that this can run without error
        model.get_fantasy_model(new_x, new_y) # gpytorch kernel causes to get fantasy_strategy in gpytorch.exact_prediction_strategies which misses QEP


if __name__ == "__main__":
    unittest.main()
