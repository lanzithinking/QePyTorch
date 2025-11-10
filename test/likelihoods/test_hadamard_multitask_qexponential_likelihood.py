#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import ToeplitzLinearOperator
from torch.distributions import Distribution

import qpytorch
from qpytorch.distributions import MultivariateQExponential

from qpytorch.likelihoods import QExponentialLikelihood, HadamardQExponentialLikelihood, Likelihood
from qpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase

POWER = 2.0

class TestMultitaskQExponentialLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 2

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5)

    def _create_task_input(self, batch_shape=torch.Size([])):
        return torch.tensor([0, 1, 2, 3, 0], dtype=torch.long).expand(*batch_shape, 5).unsqueeze(-1)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_and_task_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        return MultivariateQExponential(torch.randn(*batch_shape, 5), data_and_task_mat, power=torch.tensor(POWER))

    def create_likelihood(self):
        return HadamardQExponentialLikelihood(num_tasks=4, power=torch.tensor(POWER))

    def _test_conditional(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_conditional_input(batch_shape)
        task_idcs = self._create_task_input(batch_shape)
        output = likelihood(input, [task_idcs])

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape, input.shape)

    def _test_log_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        task_idcs = self._create_task_input(batch_shape)
        target = self._create_targets(batch_shape)
        with qpytorch.settings.num_likelihood_samples(512):
            output = likelihood.log_marginal(target, input, [task_idcs])

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))

        with qpytorch.settings.num_likelihood_samples(512):
            # Since all tasks are initialized with the same noise, this is
            # equivalent to using a shared QExponentialLikelihood
            default_log_prob = Likelihood.log_marginal(QExponentialLikelihood(power=torch.tensor(POWER)), target, input)
        self.assertAllClose(output, default_log_prob, rtol=0.25)

    def _test_log_prob(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        task_idcs = self._create_task_input(batch_shape)
        target = self._create_targets(batch_shape)
        with qpytorch.settings.num_likelihood_samples(512):
            output = likelihood.expected_log_prob(target, input, [task_idcs])

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))

        with qpytorch.settings.num_likelihood_samples(512):
            # Since all tasks are initialized with the same noise, this is
            # equivalent to using a shared QExponentialLikelihood
            default_log_prob = Likelihood.expected_log_prob(QExponentialLikelihood(power=torch.tensor(POWER)), target, input)
        self.assertAllClose(output, default_log_prob, rtol=0.25)

    def _test_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        task_idcs = self._create_task_input(batch_shape)
        output = likelihood(input, [task_idcs])

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape[-len(input.sample().shape) :], input.sample().shape)

        # Compare against default implementation
        with qpytorch.settings.num_likelihood_samples(30000):
            default = Likelihood.marginal(likelihood, input, [task_idcs])
        # print(output.mean, default.mean)
        default_mean = default.mean
        actual_mean = output.mean
        if default_mean.dim() > actual_mean.dim():
            default_mean = default_mean.mean(0)
        self.assertAllClose(default_mean, actual_mean, rtol=0.25, atol=0.25)

    def test_marginal_variance(self):
        likelihood = HadamardQExponentialLikelihood(num_tasks=4, power=torch.tensor(POWER))
        likelihood.noise = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        input = self._create_marginal_input()
        task_idcs = self._create_task_input()
        variance = likelihood(input, [task_idcs]).variance
        self.assertAllClose(variance, torch.tensor([1.1, 1.2, 1.3, 1.4, 1.1]))

        # The shape and dtypes of the tasks must be (num_data, 1) and integer dtype
        with self.assertRaises(ValueError):
            likelihood(input)

        with self.assertRaises(ValueError):
            likelihood(input, [])

        with self.assertRaises(ValueError):
            likelihood(input, [task_idcs.squeeze(-1)])

        # test with task feature and full input
        likelihood = HadamardQExponentialLikelihood(num_tasks=4, task_feature_index=1, power=torch.tensor(POWER))
        likelihood.noise = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        X = torch.cat([torch.zeros_like(task_idcs), task_idcs], dim=-1)
        self.assertEqual(variance, likelihood(input, [X]).variance)
