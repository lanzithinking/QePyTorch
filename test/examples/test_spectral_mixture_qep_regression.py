#!/usr/bin/env python3

import unittest
from collections import OrderedDict
from math import exp, pi

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import SpectralMixtureKernel
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior
from qpytorch.test import BaseTestCase

POWER = 1.0

# Simple training data: let's try to learn a sine function
train_x = torch.linspace(0, 1, 15)
train_y = torch.sin(train_x * (2 * pi))

# Spectral mixture kernel should be able to train on
# data up to x=0.75, but test on data up to x=2
test_x = torch.linspace(0, 1.5, 51)
test_y = torch.sin(test_x * (2 * pi))

good_state_dict = OrderedDict(
    [
        ("likelihood.log_noise", torch.tensor([-5.0])),
        ("mean_module.raw_constant", torch.tensor([0.4615])),
        ("covar_module.log_mixture_weights", torch.tensor([-0.7277, -15.1212, -0.5511, -6.3787]).unsqueeze(0)),
        (
            "covar_module.log_mixture_means",
            torch.tensor([[-0.1201], [0.6013], [-3.7319], [0.2380]]).unsqueeze(0).unsqueeze(-2),
        ),
        (
            "covar_module.log_mixture_scales",
            torch.tensor([[-1.9713], [2.6217], [-3.9268], [-4.7071]]).unsqueeze(0).unsqueeze(-2),
        ),
    ]
)


class SpectralMixtureQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood, empspect=False):
        super(SpectralMixtureQEPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=1)
        if empspect:
            self.covar_module.initialize_from_data(train_x, train_y)
        else:
            self.covar_module.initialize_from_data_empspect(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestSpectralMixtureQEPRegression(BaseTestCase, unittest.TestCase):
    seed = 4

    def test_spectral_mixture_qep_mean_abs_error_empspect_init(self):
        return self.test_spectral_mixture_qep_mean_abs_error(empspect=True)

    def test_spectral_mixture_qep_mean_abs_error(self, empspect=False):
        likelihood = QExponentialLikelihood(noise_prior=SmoothedBoxPrior(exp(-5), exp(3), sigma=0.1), power=torch.tensor(POWER))
        qep_model = SpectralMixtureQEPModel(train_x, train_y, likelihood, empspect=empspect)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        # Optimize the model
        qep_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(qep_model.parameters()), lr=0.01)
        optimizer.n_iter = 0

        if not empspect:
            qep_model.load_state_dict(good_state_dict, strict=False)

        for i in range(300):
            optimizer.zero_grad()
            output = qep_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

            if i == 0:
                for param in qep_model.parameters():
                    self.assertTrue(param.grad is not None)
                    # TODO: Uncomment when we figure out why this is flaky.
                    # self.assertGreater(param.grad.norm().item(), 0.)

        # Test the model
        with torch.no_grad():
            qep_model.eval()
            likelihood.eval()
            test_preds = likelihood(qep_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        # The spectral mixture kernel should be trivially able to
        # extrapolate the sine function.
        self.assertLess(mean_abs_error.squeeze().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
