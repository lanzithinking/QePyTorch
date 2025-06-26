#!/usr/bin/env python3

import unittest

import torch

import qpytorch
from qpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

def multitask_likelihood_cls():
    return qpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4) if POWER==2 \
           else qpytorch.likelihoods.MultitaskQExponentialLikelihood(num_tasks=4, power=torch.tensor(POWER))


def singletask_likelihood_cls():
    return qpytorch.likelihoods.GaussianLikelihood() if POWER==2 \
           else qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER))


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return qpytorch.variational.LMCVariationalStrategy(
        qpytorch.variational.VariationalStrategy(
            model, inducing_points, variational_distribution, learn_inducing_locations
        ),
        num_tasks=4,
        num_latents=3,
        latent_dim=-1,
    )


class TestLMCVariational(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([3])

    @property
    def event_shape(self):
        return torch.Size([32, 4])

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
        cg_mock, _, ciq_mock = super().test_training_iteration(
            *args, expected_batch_shape=expected_batch_shape, **kwargs
        )
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        cg_mock, _, ciq_mock = super().test_eval_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestLMCPredictive(TestLMCVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestLMCRobust(TestLMCVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldLMCVariational(TestLMCVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCPredictive(TestLMCPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCRobust(TestLMCRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaLMCVariational(TestLMCVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCPredictive(TestLMCPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCRobust(TestLMCRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestLMCVariationalSharedVariational(TestLMCVariational, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([3])


class TestLMCPredictiveSharedVariational(TestLMCVariationalSharedVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestLMCRobustSharedVariational(TestLMCVariationalSharedVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldLMCVariationalSharedVariational(TestLMCVariationalSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCPredictiveSharedVariational(TestLMCPredictiveSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCRobustSharedVariational(TestLMCRobustSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaLMCVariationalSharedVariational(TestLMCVariationalSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCPredictiveSharedVariational(TestLMCPredictiveSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCRobustSharedVariational(TestLMCRobustSharedVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestIndexedLMCVariational(TestLMCVariational, unittest.TestCase):
    def _training_iter(
        self, model, likelihood, batch_shape=torch.Size([]), mll_cls=qpytorch.mlls.VariationalELBO, cuda=False
    ):
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


class TestIndexedLMCPredictive(TestIndexedLMCVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.PredictiveLogLikelihood


class TestIndexedLMCRobust(TestIndexedLMCVariational):
    @property
    def mll_cls(self):
        return qpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldIndexedLMCVariational(TestIndexedLMCVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedLMCPredictive(TestIndexedLMCPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedLMCRobust(TestIndexedLMCRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaIndexedLMCVariational(TestIndexedLMCVariational):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedLMCPredictive(TestIndexedLMCPredictive):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedLMCRobust(TestIndexedLMCRobust):
    @property
    def distribution_cls(self):
        return qpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
