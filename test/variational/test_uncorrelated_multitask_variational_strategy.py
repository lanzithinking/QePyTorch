#!/usr/bin/env python3

import unittest

import torch

import qpytorch
from qpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

def multitask_likelihood_cls():
    return qpytorch.likelihoods.MultitaskQExponentialLikelihood(num_tasks=2, power=torch.tensor(POWER))


def singletask_likelihood_cls():
    return qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER))


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return qpytorch.variational.UncorrelatedMultitaskVariationalStrategy(
        qpytorch.variational.VariationalStrategy(
            model, inducing_points, variational_distribution, learn_inducing_locations
        ),
        num_tasks=2,
    )


class TestMultitaskVariationalQEP(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([2])

    @property
    def event_shape(self):
        return torch.Size([32, 2])

    @property
    def distribution_cls(self):
        return qpytorch.variational.CholeskyVariationalDistribution

    @property
    def likelihood_cls(self):
        return multitask_likelihood_cls

    @property
    def mll_cls(self):
        return qpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return strategy_cls

    def test_training_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        super().test_training_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)

    def test_eval_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        super().test_eval_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestMultitaskPredictiveQEP(TestMultitaskVariationalQEP):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestMultitaskRobustVQEP(TestMultitaskVariationalQEP):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldMultitaskVariationalQEP(TestMultitaskVariationalQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldMultitaskPredictiveQEP(TestMultitaskPredictiveQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldMultitaskRobustVQEP(TestMultitaskRobustVQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaMultitaskVariationalQEP(TestMultitaskVariationalQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaMultitaskPredictiveQEP(TestMultitaskPredictiveQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaMultitaskRobustVQEP(TestMultitaskRobustVQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestIndexedMultitaskVariationalQEP(TestMultitaskVariationalQEP, unittest.TestCase):
    def _training_iter(
        self, model, likelihood, batch_shape=torch.Size([]), mll_cls=qpytorch.mlls.VariationalELBO, cuda=False
    ):
        batch_shape = list(batch_shape)
        batch_shape[-1] = 1
        train_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        train_i = torch.rand(*batch_shape, 32).round().long()
        train_y = torch.linspace(-1, 1, self.event_shape[0])
        train_y = train_y.view(self.event_shape[0], *([1] * (len(self.event_shape) - 1)))
        train_y = train_y.expand(*self.event_shape)
        mll = mll_cls(likelihood, model, num_data=train_x.size(-2))
        if cuda:
            train_x = train_x.cuda()
            train_i = train_i.cuda()
            train_y = train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Single optimization iteration
        model.train()
        likelihood.train()
        output = model(train_x, task_indices=train_i)
        loss = -mll(output, train_y)
        loss.sum().backward()

        # Make sure we have gradients for all parameters
        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        return output, loss

    def _eval_iter(self, model, batch_shape=torch.Size([]), cuda=False):
        batch_shape = list(batch_shape)
        batch_shape[-1] = 1
        test_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        test_i = torch.rand(*batch_shape, 32).round().long()
        if cuda:
            test_x = test_x.cuda()
            test_i = test_i.cuda()
            model = model.cuda()

        # Single optimization iteration
        model.eval()
        with torch.no_grad():
            output = model(test_x, task_indices=test_i)

        return output

    @property
    def event_shape(self):
        return torch.Size([32])

    @property
    def likelihood_cls(self):
        return singletask_likelihood_cls


class TestIndexedMultitaskPredictiveQEP(TestIndexedMultitaskVariationalQEP):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestIndexedMultitaskRobustVQEP(TestIndexedMultitaskVariationalQEP):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldIndexedMultitaskVariationalQEP(TestIndexedMultitaskVariationalQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedMultitaskPredictiveQEP(TestIndexedMultitaskPredictiveQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedMultitaskRobustVQEP(TestIndexedMultitaskRobustVQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaIndexedMultitaskVariationalQEP(TestIndexedMultitaskVariationalQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedMultitaskPredictiveQEP(TestIndexedMultitaskPredictiveQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedMultitaskRobustVQEP(TestIndexedMultitaskRobustVQEP):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
