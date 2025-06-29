#!/usr/bin/env python3
# Mostly copied from https://raw.githubusercontent.com/pyro-ppl/pyro/dev/tests/distributions/test_delta.py

import unittest

import numpy as np
import torch

from qpytorch.distributions import QExponential
from qpytorch.test import BaseTestCase


class TestQExponential(BaseTestCase, unittest.TestCase):
    def setUp(self):
        self.m = torch.tensor([3.0])
        self.ms = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.ms_expanded = self.ms.expand(4, 3)
        self.v = torch.tensor([1.0])
        self.vs = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        self.vs_expanded = self.vs.expand(4, 3)
        self.q = torch.tensor(1.0)
        self.test_data = torch.tensor([[3.0], [3.0], [3.0]])
        self.batch_test_data_1 = torch.arange(0.0, 4.0).unsqueeze(1).expand(4, 3)
        self.batch_test_data_2 = torch.arange(4.0, 8.0).unsqueeze(1).expand(4, 3)
        self.batch_test_data_3 = torch.Tensor([[3.0], [3.0], [3.0], [3.0]])
        self.expected_support = [[[0.0], [1.0], [2.0], [3.0]]]
        self.expected_support_non_vec = [[3.0]]
        self.analytic_mean = 3.0
        self.analytic_var = 1.0
        self.n_samples = 100000

    def test_log_prob_sum(self):
        log_px_torch = QExponential(self.m, self.v, self.q).log_prob(self.test_data).sum()
        self.assertEqual(log_px_torch.item(), torch.inf)

    def test_batch_log_prob(self):
        log_px_torch = QExponential(self.ms_expanded, self.vs_expanded, self.q).log_prob(self.batch_test_data_1).data
        self.assertEqual(log_px_torch.sum().item(), torch.inf)
        log_px_torch = QExponential(self.ms_expanded, self.vs_expanded, self.q).log_prob(self.batch_test_data_2).data
        self.assertLess((log_px_torch.sum()+51.6628).item(), 1e-2)

    def test_batch_log_prob_shape(self):
        assert QExponential(self.ms, self.vs, self.q).log_prob(self.batch_test_data_3).size() == (4, 1)
        assert QExponential(self.m, self.v, self.q).log_prob(self.batch_test_data_3).size() == (4, 1)

    def test_mean_and_var(self):
        torch_samples = QExponential(self.m, self.v, self.q).rsample(torch.Size([self.n_samples])).detach().cpu().numpy()
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertLess((torch_mean-self.analytic_mean).item(), 2e-2)
        self.assertLess((torch_var-3*self.analytic_var).item(), 5e-2)

if __name__ == "__main__":
    unittest.main()
