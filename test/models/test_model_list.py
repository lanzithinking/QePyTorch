#!/usr/bin/env python3

import unittest

import torch

from qpytorch.likelihoods import FixedNoiseGaussianLikelihood, FixedNoiseQExponentialLikelihood
from qpytorch.models import IndependentModelList

# from test_exact_gp import TestExactGP
from .test_exact_qep import TestExactQEP

TEST_MDL = 'QEP'; POWER = {'GP': 2.0, 'QEP': 1.0}[TEST_MDL]

class TestModelList(unittest.TestCase):
    def create_model(self, fixed_noise=False):
        if TEST_MDL == 'GP':
            data = TestExactGP.create_test_data(self)
            likelihood, labels = TestExactGP.create_likelihood_and_labels(self)
            if fixed_noise:
                noise = 0.1 + 0.2 * torch.rand_like(labels)
                likelihood = FixedNoiseGaussianLikelihood(noise)
            return TestExactGP.create_model(self, data, labels, likelihood)
        elif TEST_MDL == 'QEP':
            data = TestExactQEP.create_test_data(self)
            likelihood, labels = TestExactQEP.create_likelihood_and_labels(self)
            if fixed_noise:
                noise = 0.1 + 0.2 * torch.rand_like(labels)
                likelihood = FixedNoiseQExponentialLikelihood(noise)
            return TestExactQEP.create_model(self, data, labels, likelihood)

    def test_forward_eval(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        with self.assertRaises(ValueError):
            model(torch.rand(3))
        model(torch.rand(3), torch.rand(3))

    def test_forward_eval_fixed_noise(self):
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3), torch.rand(3))

    def test_get_fantasy_model(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3), torch.rand(3))
        fant_x = [torch.randn(2), torch.randn(3)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y)
        fmodel(torch.randn(4), torch.randn(4))

    def test_get_fantasy_model_fixed_noise(self):
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3), torch.rand(3))
        fant_x = [torch.randn(2), torch.randn(3)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fant_noise = [0.1 * torch.ones(2), 0.1 * torch.ones(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y, noise=fant_noise)
        fmodel(torch.randn(4), torch.randn(4))


if __name__ == "__main__":
    unittest.main()
