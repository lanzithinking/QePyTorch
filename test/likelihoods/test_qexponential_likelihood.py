#!/usr/bin/env python3

import pickle
import unittest

import torch
from linear_operator.operators import DiagLinearOperator

from qpytorch import settings
from qpytorch.distributions import MultivariateQExponential
from qpytorch.likelihoods import FixedNoiseQExponentialLikelihood, QExponentialLikelihood
from qpytorch.likelihoods.qexponential_likelihood import QExponentialDirichletClassificationLikelihood
from qpytorch.likelihoods.noise_models import FixedNoise
from qpytorch.priors import GammaPrior
from qpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase

POWER = 2.0

class TestQExponentialLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0; _power = POWER

    def create_likelihood(self):
        return QExponentialLikelihood(power=torch.tensor(POWER))

    def test_pickle_with_prior(self):
        likelihood = QExponentialLikelihood(noise_prior=GammaPrior(1, 1), power=torch.tensor(POWER))
        pickle.loads(pickle.dumps(likelihood))  # Should be able to pickle and unpickle with a prior


class TestQExponentialLikelihoodBatch(TestQExponentialLikelihood):
    seed = 0; _power = POWER

    def create_likelihood(self):
        return QExponentialLikelihood(batch_shape=torch.Size([3]), power=torch.tensor(POWER))

    def test_nonbatch(self):
        pass


class TestQExponentialLikelihoodMultiBatch(TestQExponentialLikelihood):
    seed = 0; _power = POWER

    def create_likelihood(self):
        return QExponentialLikelihood(batch_shape=torch.Size([2, 3]), power=torch.tensor(POWER))

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


class TestFixedNoiseQExponentialLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    _power = POWER
    def create_likelihood(self):
        noise = 0.1 + torch.rand(5)
        power = torch.tensor(POWER)
        return FixedNoiseQExponentialLikelihood(noise=noise, power=power)

    def test_fixed_noise_qexponential_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            power = torch.tensor(POWER, device=device, dtype=dtype)
            lkhd = FixedNoiseQExponentialLikelihood(noise=noise, power=power)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedNoise)
            self.assertTrue(torch.equal(noise, lkhd.noise))
            new_noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(4, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(4, device=device, dtype=dtype))
            qep = MultivariateQExponential(mean, covar, power)
            out = lkhd(qep)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(5, device=device, dtype=dtype))
            qep = MultivariateQExponential(mean, covar, power)
            with self.assertWarns(UserWarning):
                lkhd(qep)
            # test __call__ w/ observation noise
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            out = lkhd(qep, noise=obs_noise)
            self.assertTrue(torch.allclose(out.variance, 1 + obs_noise))
            # test noise smaller than min_fixed_noise
            expected_min_noise = settings.min_fixed_noise.value(dtype)
            noise[:2] = 0
            lkhd = FixedNoiseQExponentialLikelihood(noise=noise, power=power)
            expected_noise = noise.clone()
            expected_noise[:2] = expected_min_noise
            self.assertTrue(torch.allclose(lkhd.noise, expected_noise))


class TestFixedNoiseQExponentialLikelihoodBatch(BaseLikelihoodTestCase, unittest.TestCase):
    _power = POWER
    def create_likelihood(self):
        noise = 0.1 + torch.rand(3, 5)
        power = torch.tensor(POWER)
        return FixedNoiseQExponentialLikelihood(noise=noise, power=power)

    def test_nonbatch(self):
        pass


class TestFixedNoiseQExponentialLikelihoodMultiBatch(BaseLikelihoodTestCase, unittest.TestCase):
    _power = POWER
    def create_likelihood(self):
        noise = 0.1 + torch.rand(2, 3, 5)
        power = torch.tensor(POWER)
        return FixedNoiseQExponentialLikelihood(noise=noise, power=power)

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


class TestDirichletClassificationLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    _power = POWER
    def create_likelihood(self):
        train_x = torch.randn(15)
        labels = torch.round(train_x).long()
        power = torch.tensor(POWER)
        likelihood = QExponentialDirichletClassificationLikelihood(labels, power=power)
        return likelihood

    def test_batch(self):
        pass

    def test_multi_batch(self):
        pass

    def test_nonbatch(self):
        pass

    def test_dirichlet_classification_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            power = torch.tensor(POWER, device=device, dtype=dtype)
            lkhd = QExponentialDirichletClassificationLikelihood(noise, dtype=dtype, power=power)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedNoise)
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            new_noise, _, _ = lkhd._prepare_targets(noise, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(6, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(6, device=device, dtype=dtype))
            qep = MultivariateQExponential(mean, covar, power)
            out = lkhd(qep)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(5, device=device, dtype=dtype))
            qep = MultivariateQExponential(mean, covar, power)
            with self.assertWarns(UserWarning):
                lkhd(qep)
            # test __call__ w/ new targets
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            obs_noise = (obs_noise > 0.5).long()
            out = lkhd(qep, targets=obs_noise)
            obs_targets, _, _ = lkhd._prepare_targets(obs_noise, dtype=dtype)
            self.assertTrue(torch.allclose(out.variance, 1.0 + obs_targets))


class TestQExponentialLikelihoodWithMissingObs(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 42; _power = POWER

    def create_likelihood(self):
        return QExponentialLikelihood(power=torch.tensor(POWER))

    def test_missing_value_inference_fill(self):
        """
        samples = qep samples + noise samples
        In this test, we try to recover noise parameters when some elements in
        'samples' are missing at random.
        """

        torch.manual_seed(self.seed)

        qep, samples = self._make_data()

        missing_probability = 0.33
        missing_idx = torch.distributions.Binomial(1, missing_probability).sample(samples.shape).bool()
        samples[missing_idx] = float("nan")

        # check that the correct noise sd is recovered

        with settings.observation_nan_policy("fill"):
            self._check_recovery(qep, samples)

    def test_missing_value_inference_mask(self):
        """
        samples = qep samples + noise samples
        In this test, we try to recover noise parameters when some elements in
        'samples' are missing at random.
        """

        torch.manual_seed(self.seed)

        qep, samples = self._make_data()

        missing_prop = 0.33
        missing_idx = torch.distributions.Binomial(1, missing_prop).sample(samples.shape[1:]).bool()
        samples[1, missing_idx] = float("nan")

        # check that the correct noise sd is recovered

        with settings.observation_nan_policy("fill"):
            self._check_recovery(qep, samples)

    def _make_data(self):
        mu = torch.zeros(2, 3)
        sigma = torch.tensor([[[1, 0.999, -0.999], [0.999, 1, -0.999], [-0.999, -0.999, 1]]] * 2).float()
        power = torch.tensor(POWER)
        qep = MultivariateQExponential(mu, sigma, power)
        samples = qep.sample(torch.Size([10000]))  # qep samples
        noise_sd = 0.5
        noise_dist = torch.distributions.Normal(0, noise_sd)
        samples += noise_dist.sample(samples.shape)  # noise
        return qep, samples

    def _check_recovery(self, qep, samples):
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        opt = torch.optim.Adam(likelihood.parameters(), lr=0.05)
        for _ in range(100):
            opt.zero_grad()
            loss = -likelihood.log_marginal(samples, qep).sum()
            loss.backward()
            opt.step()
        self.assertTrue(abs(float(likelihood.noise.sqrt()) - 0.5) < 0.025)
        # Check log marginal works
        likelihood.log_marginal(samples[0], qep)


if __name__ == "__main__":
    unittest.main()
