#!/usr/bin/env python3

import math
import unittest

import torch

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import RBFKernel, ScaleKernel
from qpytorch.likelihoods import QExponentialLikelihood, LikelihoodList
from qpytorch.means import ConstantMean
from qpytorch.mlls import SumMarginalLogLikelihood
from qpytorch.models import IndependentModelList
from qpytorch.priors import SmoothedBoxPrior
from gpytorch.test.utils import least_used_cuda_device

POWER = 1.0

class ExactQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactQEPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=SmoothedBoxPrior(math.exp(-3), math.exp(3), sigma=0.1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestModelListQEPRegression(unittest.TestCase):
    def test_simple_model_list_qep_regression(self, cuda=False):
        train_x1 = torch.linspace(0, 0.95, 25) + 0.05 * torch.rand(25)
        train_x2 = torch.linspace(0, 0.95, 15) + 0.05 * torch.rand(15)

        train_y1 = torch.sin(train_x1 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x1)
        train_y2 = torch.cos(train_x2 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x2)

        likelihood1 = QExponentialLikelihood(power=torch.tensor(POWER))
        model1 = ExactQEPModel(train_x1, train_y1, likelihood1)

        likelihood2 = QExponentialLikelihood(power=torch.tensor(POWER))
        model2 = ExactQEPModel(train_x2, train_y2, likelihood2)

        model = IndependentModelList(model1, model2)
        likelihood = LikelihoodList(model1.likelihood, model2.likelihood)

        if cuda:
            model = model.cuda()

        model.train()
        likelihood.train()

        mll = SumMarginalLogLikelihood(likelihood, model)

        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), qpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 10, device=torch.device("cuda") if cuda else torch.device("cpu"))
            outputs_f = model(test_x, test_x)
            predictions_obs_noise = likelihood(*outputs_f)

        self.assertIsInstance(outputs_f, list)
        self.assertEqual(len(outputs_f), 2)
        self.assertIsInstance(predictions_obs_noise, list)
        self.assertEqual(len(predictions_obs_noise), 2)

    def test_simple_model_list_qep_regression_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_simple_model_list_qep_regression(cuda=True)


if __name__ == "__main__":
    unittest.main()
