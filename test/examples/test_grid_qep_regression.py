#!/usr/bin/env python3

import math
import os
import random
import unittest
import warnings

import torch
from torch import optim

import qpytorch
from gpytorch.test.utils import least_used_cuda_device
from qpytorch.utils.warnings import QEPInputWarning

POWER = 1.0

def make_data(grid, cuda=False):
    train_x = qpytorch.utils.grid.create_data_from_grid(grid)
    train_y = torch.sin((train_x.sum(-1)) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
    n = 20
    test_x = torch.zeros(int(pow(n, 2)), 2)
    for i in range(n):
        for j in range(n):
            test_x[i * n + j][0] = float(i) / (n - 1)
            test_x[i * n + j][1] = float(j) / (n - 1)
    test_y = torch.sin(((test_x.sum(-1)) * (2 * math.pi)))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class GridQEPRegressionModel(qpytorch.models.ExactQEP):
    def __init__(self, grid, train_x, train_y, likelihood):
        super(GridQEPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = qpytorch.means.ConstantMean()
        self.covar_module = qpytorch.kernels.GridKernel(qpytorch.kernels.RBFKernel(), grid=grid)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestGridQEPRegression(unittest.TestCase):
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

    def test_grid_qep_mean_abs_error(self, num_dim=1, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        grid_bounds = [(0, 1)] if num_dim == 1 else [(0, 1), (0, 2)]
        grid_size = 25
        grid = torch.zeros(grid_size, len(grid_bounds), device=device)
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(
                grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size, device=device
            )

        train_x, train_y, test_x, test_y = make_data(grid, cuda=cuda)
        likelihood = qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = GridQEPRegressionModel(grid, train_x, train_y, likelihood)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        if cuda:
            qep_model.cuda()
            likelihood.cuda()

        # Optimize the model
        qep_model.train()
        likelihood.train()

        optimizer = optim.Adam(qep_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        with qpytorch.settings.debug(True):
            for _ in range(20):
                optimizer.zero_grad()
                output = qep_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for name, param in qep_model.named_parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            qep_model.eval()
            likelihood.eval()
            # Make sure we don't get QEP input warnings for testing on training data
            warnings.simplefilter("ignore", QEPInputWarning)

            train_preds = likelihood(qep_model(train_x)).mean
            mean_abs_error = torch.mean(torch.abs(train_y - train_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.3)

    def test_grid_qep_mean_abs_error_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_grid_qep_mean_abs_error(cuda=True)

    def test_grid_qep_mean_abs_error_2d(self):
        self.test_grid_qep_mean_abs_error(num_dim=2)

    def test_grid_qep_mean_abs_error_2d_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_grid_qep_mean_abs_error(cuda=True, num_dim=2)


if __name__ == "__main__":
    unittest.main()
