#!/usr/bin/env python3

import unittest

import torch

import qpytorch
from qpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

class TestCiqVariational(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return qpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return qpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return qpytorch.variational.CiqVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 4)  # One for each forward pass, one for each backward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 2)  # One for each evaluation call

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestMeanFieldCiqVariational(TestCiqVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaCiqVariational(TestCiqVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestNgdCiqVariational(TestCiqVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.NaturalVariationalDistribution


if __name__ == "__main__":
    unittest.main()
