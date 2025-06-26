#!/usr/bin/env python3

import os
import random
import unittest
import warnings
from math import exp, pi

import torch
from torch import optim

import qpytorch
from qpytorch.distributions import MultivariateQExponential
from qpytorch.kernels import GridInterpolationKernel, RBFKernel, ScaleKernel
from qpytorch.likelihoods import FixedNoiseQExponentialLikelihood
from qpytorch.means import ConstantMean
from qpytorch.priors import SmoothedBoxPrior
from gpytorch.test.utils import least_used_cuda_device
from qpytorch.utils.warnings import QEPInputWarning

POWER = 1.0

# Simple training data: let's try to learn a sine function,
# but with KISS-QEP let's use 100 training examples.
def make_data(cuda=False):
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * pi))
    test_x = torch.linspace(0, 1, 51)
    test_y = torch.sin(test_x * (2 * pi))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class QEPRegressionModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super(QEPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)))
        self.grid_covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=50, num_dims=1)
        self.covar_module = self.grid_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestKISSQEPWhiteNoiseRegression(unittest.TestCase):
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

    def test_kissqep_qep_mean_abs_error(self):
        # This test throws a warning because the fixed noise likelihood gets the wrong input
        warnings.simplefilter("ignore", QEPInputWarning)

        train_x, train_y, test_x, test_y = make_data()
        likelihood = FixedNoiseQExponentialLikelihood(torch.ones(100) * 0.001, power=torch.tensor(POWER))
        qep_model = QEPRegressionModel(train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        # Optimize the model
        qep_model.train()
        likelihood.train()

        optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        with qpytorch.settings.debug(False):
            for _ in range(25):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            qep_model.eval()
            likelihood.eval()

            test_preds = likelihood(qep_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)

    def test_kissqep_qep_fast_pred_var(self):
        with qpytorch.settings.fast_pred_var(), qpytorch.settings.debug(False):
            train_x, train_y, test_x, test_y = make_data()
            likelihood = FixedNoiseQExponentialLikelihood(torch.ones(100) * 0.001, power=torch.tensor(POWER))
            qep_model = QEPRegressionModel(train_x, train_y, likelihood)
            mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

            # Optimize the model
            qep_model.train()
            likelihood.train()

            optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(25):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in qep_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            qep_model.eval()
            likelihood.eval()
            # Set the cache
            test_function_predictions = likelihood(qep_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.noise = torch.ones(100) * 3.0
            test_function_predictions = likelihood(qep_model(train_x))

            noise = likelihood.noise
            var_diff = (test_function_predictions.variance - noise).abs()
            self.assertLess(torch.max(var_diff / noise), 0.05)

    def test_kissqep_qep_mean_abs_error_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, train_y, test_x, test_y = make_data(cuda=True)
            likelihood = FixedNoiseQExponentialLikelihood(torch.ones(100) * 0.001, power=torch.tensor(POWER)).cuda()
            qep_model = QEPRegressionModel(train_x, train_y, likelihood).cuda()
            mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

            # Optimize the model
            qep_model.train()
            likelihood.train()

            optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
            optimizer.n_iter = 0
            with qpytorch.settings.debug(False):
                for _ in range(25):
                    optimizer.zero_grad()
                    output = qep_model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.n_iter += 1
                    optimizer.step()

                for param in qep_model.parameters():
                    self.assertTrue(param.grad is not None)
                    self.assertGreater(param.grad.norm().item(), 0)

                # Test the model
                qep_model.eval()
                likelihood.eval()
                test_preds = likelihood(qep_model(test_x)).mean
                mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

            self.assertLess(mean_abs_error.squeeze().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
