#!/usr/bin/env python3

import unittest
from math import exp, pi

import torch
from torch import optim

import gpytorch, qpytorch
from qpytorch.distributions import MultivariateNormal, MultivariateQExponential
from qpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from qpytorch.likelihoods import FixedNoiseQExponentialLikelihood
from qpytorch.likelihoods.noise_models import FixedNoise
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior
from qpytorch.test import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device


POWER = 1.0

class ExactModel(gpytorch.models.ExactGP if POWER==2 else qpytorch.models.ExactQEP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x) if POWER==2 else MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestFixedNoiseFantasies(BaseTestCase, unittest.TestCase):
    seed = 1

    def _get_data(self, cuda=False, num_data=11, add_noise=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # Simple training data: let's try to learn a sine function
        train_x = torch.linspace(0, 1, num_data, device=device)
        train_y = torch.sin(train_x * (2 * pi))
        if add_noise:
            train_y.add_(torch.randn_like(train_x).mul_(0.1))
        test_x = torch.linspace(0, 1, 51, device=device)
        test_y = torch.sin(test_x * (2 * pi))
        return train_x, test_x, train_y, test_y

    def test_fixed_noise_fantasy_updates_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fixed_noise_fantasy_updates(cuda=True)

    def test_fixed_noise_fantasy_updates(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        noise = torch.full_like(train_y, 2e-4)
        test_noise = torch.full_like(test_y, 3e-4)

        likelihood = FixedNoiseGaussianLikelihood(noise) if POWER==2 else FixedNoiseQExponentialLikelihood(noise, power=torch.tensor(POWER))
        xt_model = ExactModel(train_x, train_y, likelihood)
        mll = (gpytorch if POWER==2 else qpytorch).ExactMarginalLogLikelihood(likelihood, xt_model)
        xt_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        xt_model.mean_module.initialize(constant=0)

        if cuda:
            xt_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        xt_model.train()
        likelihood.train()
        optimizer = optim.Adam(xt_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with qpytorch.settings.debug(False):
                output = xt_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in xt_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        train_x.requires_grad = True
        xt_model.set_train_data(train_x, train_y)
        with qpytorch.settings.fast_pred_var(), qpytorch.settings.detach_test_caches(False):
            # Test the model
            xt_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(xt_model(test_x), noise=test_noise)
            test_function_predictions.mean.sum().backward()

            real_fant_x_grad = train_x.grad[5:].clone()
            train_x.grad = None
            train_x.requires_grad = False
            xt_model.set_train_data(train_x, train_y)

            # Cut data down, and then add back via the fantasy interface
            xt_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            xt_model.likelihood.noise_covar = FixedNoise(noise=noise[:5])
            likelihood(xt_model(test_x), noise=test_noise)

            fantasy_x = train_x[5:].clone().detach().requires_grad_(True)
            fant_model = xt_model.get_fantasy_model(fantasy_x, train_y[5:], noise=noise[5:])

            fant_function_predictions = likelihood(fant_model(test_x), noise=test_noise)

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean, atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)

            relative_error = torch.norm(real_fant_x_grad - fantasy_x.grad) / fantasy_x.grad.norm()
            self.assertLess(relative_error, 15e-1)  # This was only passing by a hair before

    def test_fixed_noise_fantasy_updates_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fixed_noise_fantasy_updates_batch(cuda=True)

    def test_fixed_noise_fantasy_updates_batch(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        noise = torch.full_like(train_y, 2e-4)
        test_noise = torch.full_like(test_y, 3e-4)

        likelihood = FixedNoiseGaussianLikelihood(noise) if POWER==2 else FixedNoiseQExponentialLikelihood(noise, power=torch.tensor(POWER))
        xt_model = ExactModel(train_x, train_y, likelihood)
        mll = (gpytorch if POWER==2 else qpytorch).ExactMarginalLogLikelihood(likelihood, xt_model)
        xt_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        xt_model.mean_module.initialize(constant=0)

        if cuda:
            xt_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        xt_model.train()
        likelihood.train()
        optimizer = optim.Adam(xt_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with qpytorch.settings.debug(False):
                output = xt_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in xt_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        with qpytorch.settings.fast_pred_var():
            # Test the model
            xt_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(xt_model(test_x), noise=test_noise)

            # Cut data down, and then add back via the fantasy interface
            xt_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            xt_model.likelihood.noise_covar = FixedNoise(noise=noise[:5])
            likelihood(xt_model(test_x), noise=test_noise)

            fantasy_x = train_x[5:].clone().unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1).requires_grad_(True)
            fantasy_y = train_y[5:].unsqueeze(0).repeat(3, 1)
            fant_model = xt_model.get_fantasy_model(fantasy_x, fantasy_y, noise=noise[5:].unsqueeze(0).repeat(3, 1))
            fant_function_predictions = likelihood(fant_model(test_x), noise=test_noise)

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean[0], atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)


if __name__ == "__main__":
    unittest.main()
