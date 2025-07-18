#!/usr/bin/env python3

import unittest

import torch

from qpytorch.distributions import QExponential
from qpytorch.priors import QExponentialPrior
from gpytorch.test.utils import least_used_cuda_device


class TestQExponentialPrior(unittest.TestCase):
    def test_qexponential_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = QExponentialPrior(0, 1, 1.0).cuda()
            self.assertEqual(prior.loc.device.type, "cuda")
            self.assertEqual(prior.scale.device.type, "cuda")

    def test_qexponential_prior_validate_args(self):
        with self.assertRaises(ValueError):
            QExponentialPrior(0, -1, torch.tensor(1.0), validate_args=True)

    def test_qexponential_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor(0.0, device=device)
        variance = torch.tensor(1.0, device=device)
        power = torch.tensor(1.0, device=device)
        prior = QExponentialPrior(mean, variance, power)
        dist = QExponential(mean, variance, power)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

    def test_qexponential_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_qexponential_prior_log_prob(cuda=True)

    def test_qexponential_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor(0.0, device=device)
        variance = torch.tensor(1.0, device=device)
        power = torch.tensor(1.0, device=device)
        prior = QExponentialPrior(mean, variance, power, transform=torch.exp)
        dist = QExponential(mean, variance, power)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))

    def test_qexponential_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_qexponential_prior_log_prob_log_transform(cuda=True)

    def test_qexponential_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        mean = torch.tensor([0.0, 1.0], device=device)
        variance = torch.tensor([1.0, 2.0], device=device)
        power = torch.tensor(1.0, device=device)
        prior = QExponentialPrior(mean, variance, power)
        dist = QExponential(mean, variance, power)
        t = torch.zeros(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.zeros(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(3, device=device))

        mean = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], device=device)
        variance = torch.tensor([[1.0, 2.0], [0.5, 1.0]], device=device)
        prior = QExponentialPrior(mean, variance, power)
        dist = QExponential(mean, variance, power)
        t = torch.zeros(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.zeros(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(3, device=device))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(2, 3, device=device))

    def test_qexponential_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_qexponential_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
