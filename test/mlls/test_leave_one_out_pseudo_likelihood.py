#!/usr/bin/env python3

import math
import unittest

import torch

from qpytorch.distributions import MultivariateNormal, MultivariateQExponential
from qpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from qpytorch.likelihoods import QExponentialLikelihood
from qpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from qpytorch.models import ExactQEP

TEST_MDL = 'QEP'
mlls = {'GP': 'gpytorch.mlls', 'QEP': 'qpytorch.mlls'}[TEST_MDL]
exec(f"from {mlls} import {'LeaveOneOutPseudoLikelihood'} ")

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=train_x.shape[:-2])
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactQEPModel(ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=train_x.shape[:-2])
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, self.likelihood.power)


class TestLeaveOneOutPseudoLikelihood(unittest.TestCase):
    def get_data(self, shapes, dtype=None, device=None):
        train_x = torch.rand(*shapes, dtype=dtype, device=device, requires_grad=True)
        train_y = torch.sin(train_x[..., 0]) + torch.cos(train_x[..., 1])
        power = torch.tensor(1.0, dtype=dtype, device=device)
        if TEST_MDL == 'GP':
            likelihood = GaussianLikelihood().to(dtype=dtype, device=device)
            model = ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype, device=device)
        elif TEST_MDL == 'QEP':
            likelihood = QExponentialLikelihood(power=power).to(dtype=dtype, device=device)
            model = ExactQEPModel(train_x, train_y, likelihood).to(dtype=dtype, device=device)
        loocv = LeaveOneOutPseudoLikelihood(likelihood=likelihood, model=model)
        return train_x, train_y, loocv

    def test_smoke(self):
        """Make sure the loocv works without batching."""
        train_x, train_y, loocv = self.get_data([5, 2])
        output = loocv.model(train_x)
        loss = -loocv(output, train_y)
        loss.backward()
        self.assertTrue(train_x.grad is not None)

    def test_smoke_batch(self):
        """Make sure the loocv works without batching."""
        train_x, train_y, loocv = self.get_data([3, 3, 3, 5, 2])
        output = loocv.model(train_x)
        loss = -loocv(output, train_y)
        assert loss.shape == (3, 3, 3)
        loss.sum().backward()
        self.assertTrue(train_x.grad is not None)

    def test_check_bordered_system(self):
        """Make sure that the bordered system solves match the naive solution."""
        n = 5
        # Compute the pseudo-likelihood via the bordered systems in O(n^3)
        train_x, train_y, loocv = self.get_data([n, 2], dtype=torch.float64)
        output = loocv.model(train_x)
        loocv_1 = loocv(output, train_y)

        # Compute the pseudo-likelihood by fitting n independent models O(n^4)
        loocv_2 = 0.0
        for i in range(n):
            inds = torch.cat((torch.arange(0, i), torch.arange(i + 1, n)))
            if TEST_MDL == 'GP':
                power = torch.tensor(2.0)
                likelihood = GaussianLikelihood()
                model = ExactGPModel(train_x[inds, :], train_y[inds], likelihood)
            elif TEST_MDL == 'QEP':
                power = torch.tensor(1.0)
                likelihood = QExponentialLikelihood(power=power)
                model = ExactQEPModel(train_x[inds, :], train_y[inds], likelihood)
            model.eval()
            with torch.no_grad():
                preds = likelihood(model(train_x[i, :].unsqueeze(0)))
                mean, var = preds.mean, preds.variance
                loocv_2 += -0.5 * var.log() - 0.5 * (train_y[i] - mean).abs().pow(power) / var**(power/2.) - 0.5 * math.log(2 * math.pi)
                if TEST_MDL == 'QEP' and power!=2: loocv_2 += (power/2.-1) * ( (train_y[i] - mean).abs().log() -0.5 * var.log() )
        loocv_2 /= n

        self.assertAlmostEqual(
            loocv_1.item(),
            loocv_2.item(),
        )


if __name__ == "__main__":
    unittest.main()
