#!/usr/bin/env python3

import torch
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.operators import (
    BlockDiagLinearOperator,
    BlockInterleavedLinearOperator,
    CatLinearOperator,
    DiagLinearOperator,
)

from .multivariate_qexponential import MultivariateQExponential


class MultitaskMultivariateQExponential(MultivariateQExponential):
    """
    Constructs a multi-output multivariate Q-Exponential random variable, based on mean and covariance
    Can be multi-output multivariate, or a batch of multi-output multivariate Q-Exponential

    Passing a matrix mean corresponds to a multi-output multivariate Q-Exponential
    Passing a matrix mean corresponds to a batch of multivariate Q-Exponentials

    :param torch.Tensor mean:  An `n x t` or batch `b x n x t` matrix of means for the QEP distribution.
    :param ~linear_operator.operators.LinearOperator covar: An `... x NT x NT` (batch) matrix.
        covariance matrix of QEP distribution.
    :param power: (default=2.0) (scalar) power of QEP distribution.
    :param bool validate_args: (default=False) If True, validate `mean` and `covariance_matrix` arguments.
    :param bool interleaved: (default=True) If True, covariance matrix is interpreted as block-diagonal w.r.t.
        inter-task covariances for each observation. If False, it is interpreted as block-diagonal
        w.r.t. inter-observation covariance for each task.
    """

    def __init__(self, mean, covariance_matrix, power=torch.tensor(2.0), validate_args=False, interleaved=True):
        if not torch.is_tensor(mean) and not isinstance(mean, LinearOperator):
            raise RuntimeError("The mean of a MultitaskMultivariateQExponential must be a Tensor or LinearOperator")

        if not torch.is_tensor(covariance_matrix) and not isinstance(covariance_matrix, LinearOperator):
            raise RuntimeError("The covariance of a MultitaskMultivariateQExponential must be a Tensor or LinearOperator")

        if mean.dim() < 2:
            raise RuntimeError("mean should be a matrix or a batch matrix (batch mode)")

        # Ensure that shapes are broadcasted appropriately across the mean and covariance
        # Means can have singleton dimensions for either the `n` or `t` dimensions
        batch_shape = torch.broadcast_shapes(mean.shape[:-2], covariance_matrix.shape[:-2])
        if mean.shape[-2:].numel() != covariance_matrix.size(-1):
            if covariance_matrix.size(-1) % mean.shape[-2:].numel():
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
            elif mean.size(-2) == 1:
                mean = mean.expand(*batch_shape, covariance_matrix.size(-1) // mean.size(-1), mean.size(-1))
            elif mean.size(-1) == 1:
                mean = mean.expand(*batch_shape, mean.size(-2), covariance_matrix.size(-2) // mean.size(-2))
            else:
                raise RuntimeError(
                    f"mean shape {mean.shape} is incompatible with covariance shape {covariance_matrix.shape}"
                )
        else:
            mean = mean.expand(*batch_shape, *mean.shape[-2:])

        self._output_shape = mean.shape
        # TODO: Instead of transpose / view operations, use a PermutationLinearOperator (see #539)
        # to handle interleaving
        self._interleaved = interleaved
        if self._interleaved:
            mean_qep = mean.reshape(*mean.shape[:-2], -1)
        else:
            mean_qep = mean.transpose(-1, -2).reshape(*mean.shape[:-2], -1)
        super().__init__(mean=mean_qep, covariance_matrix=covariance_matrix, power=power, validate_args=validate_args)

    @property
    def base_sample_shape(self):
        """
        Returns the shape of a base sample (without batching) that is used to
        generate a single sample.
        """
        base_sample_shape = self.event_shape
        return base_sample_shape

    @property
    def event_shape(self):
        return self._output_shape[-2:]

    @classmethod
    def from_batch_qep(cls, batch_qep, task_dim=-1):
        """
        Reinterpret a batch of multivariate q-exponential distributions as an (uncorrelated) multitask multivariate q-exponential
        distribution.

        :param ~qpytorch.distributions.MultivariateQExponential batch_qep: The base QEP distribution.
            (This distribution should have at least one batch dimension).
        :param int task_dim: Which batch dimension should be interpreted as the dimension for the independent tasks.
        :returns: the uncorrelated multitask distribution
        :rtype: qpytorch.distributions.MultitaskMultivariateQExponential

        Example:
            >>> # model is a qpytorch.models.VariationalQEP
            >>> # likelihood is a qpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 2, 3)
            >>> covar_factor = torch.randn(4, 2, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> power = torch.tensor(1.0)
            >>> qep = qpytorch.distributions.MultivariateQExponential(mean, covar, power)
            >>> print(qep.event_shape, qep.batch_shape)
            >>> # torch.Size([3]), torch.Size([4, 2])
            >>>
            >>> mqep = MultitaskMultivariateQExponential.from_batch_qep(qep, task_dim=-1)
            >>> print(mqep.event_shape, mqep.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
        orig_task_dim = task_dim
        task_dim = task_dim if task_dim >= 0 else (len(batch_qep.batch_shape) + task_dim)
        if task_dim < 0 or task_dim > len(batch_qep.batch_shape):
            raise ValueError(
                f"task_dim of {orig_task_dim} is incompatible with QEP batch shape of {batch_qep.batch_shape}"
            )

        num_dim = batch_qep.mean.dim()
        res = cls(
            mean=batch_qep.mean.permute(*range(0, task_dim), *range(task_dim + 1, num_dim), task_dim),
            covariance_matrix=BlockInterleavedLinearOperator(batch_qep.lazy_covariance_matrix, block_dim=task_dim),
            power=batch_qep.power
        )
        return res

    @classmethod
    def from_uncorrelated_qeps(cls, qeps):
        """
        Convert an iterable of QEPs into a :obj:`~qpytorch.distributions.MultitaskMultivariateQExponential`.
        The resulting distribution will have ``len(qeps)`` tasks, and the tasks will be uncorrelated.

        :param ~qpytorch.distributions.MultivariateQExponential qep: The base QEP distributions.
        :returns: the uncorrelated multitask distribution
        :rtype: qpytorch.distributions.MultitaskMultivariateQExponential

        Example:
            >>> # model is a qpytorch.models.VariationalQEP
            >>> # likelihood is a qpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> power = torch.tensor(1.0)
            >>> qep1 = qpytorch.distributions.MultivariateQExponential(mean, covar, power)
            >>>
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> qep2 = qpytorch.distributions.MultivariateQExponential(mean, covar, power)
            >>>
            >>> mqep = MultitaskMultivariateQExponential.from_uncorrelated_qeps([qep1, qep2])
            >>> print(mqep.event_shape, mqep.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
        if len(qeps) < 2:
            raise ValueError("Must provide at least 2 QEPs to form a MultitaskMultivariateQExponential")
        if any(isinstance(qep, MultitaskMultivariateQExponential) for qep in qeps):
            raise ValueError("Cannot accept MultitaskMultivariateQExponentials")
        if not all(m.batch_shape == qeps[0].batch_shape for m in qeps[1:]):
            batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in qeps))
            qeps = [qep.expand(batch_shape) for qep in qeps]
        if not all(m.event_shape == qeps[0].event_shape for m in qeps[1:]):
            raise ValueError("All MultivariateQExponentials must have the same event shape")
        mean = torch.stack([qep.mean for qep in qeps], -1)
        # TODO: To do the following efficiently, we don't want to evaluate the
        # covariance matrices. Instead, we want to use the lazies directly in the
        # BlockDiagLinearOperator. This will require implementing a new BatchLinearOperator:

        # https://github.com/cornellius-gp/gpytorch/issues/468
        covar_blocks_lazy = CatLinearOperator(
            *[qep.lazy_covariance_matrix.unsqueeze(0) for qep in qeps], dim=0, output_device=mean.device
        )
        covar_lazy = BlockDiagLinearOperator(covar_blocks_lazy, block_dim=0)
        return cls(mean=mean, covariance_matrix=covar_lazy, power=qeps[0].power, interleaved=False)

    @classmethod
    def from_repeated_qep(cls, qep, num_tasks):
        """
        Convert a single QEP into a :obj:`~qpytorch.distributions.MultitaskMultivariateQExponential`,
        where each task shares the same mean and covariance.

        :param ~qpytorch.distributions.MultivariateQExponential qep: The base QEP distribution.
        :param int num_tasks: How many tasks to create.
        :returns: the uncorrelated multitask distribution
        :rtype: qpytorch.distributions.MultitaskMultivariateQExponential

        Example:
            >>> # model is a qpytorch.models.VariationalQEP
            >>> # likelihood is a qpytorch.likelihoods.Likelihood
            >>> mean = torch.randn(4, 3)
            >>> covar_factor = torch.randn(4, 3, 3)
            >>> covar = covar_factor @ covar_factor.transpose(-1, -2)
            >>> qep = qpytorch.distributions.MultivariateQExponential(mean, covar)
            >>> print(qep.event_shape, qep.batch_shape)
            >>> # torch.Size([3]), torch.Size([4])
            >>>
            >>> mqep = MultitaskMultivariateQExponential.from_repeated_qep(qep, num_tasks=2)
            >>> print(mqep.event_shape, mqep.batch_shape)
            >>> # torch.Size([3, 2]), torch.Size([4])
        """
        return cls.from_batch_qep(qep.expand(torch.Size([num_tasks]) + qep.batch_shape), task_dim=0)

    def expand(self, batch_size):
        new_mean = self.mean.expand(torch.Size(batch_size) + self.mean.shape[-2:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(new_mean, new_covar, power=self.power, interleaved=self._interleaved)
        return res

    def get_base_samples(self, sample_shape=torch.Size(), **kwargs):
        base_samples = super().get_base_samples(sample_shape, **kwargs)
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = sample_shape + self._output_shape[:-2] + self._output_shape[:-3:-1]
            return base_samples.view(new_shape).transpose(-1, -2).contiguous()
        return base_samples.view(*sample_shape, *self._output_shape)

    def log_prob(self, value):
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = value.shape[:-2] + value.shape[:-3:-1]
            value = value.view(new_shape).transpose(-1, -2).contiguous()
        return super().log_prob(value.reshape(*value.shape[:-2], -1))

    @property
    def mean(self):
        mean = super().mean
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return mean.view(new_shape).transpose(-1, -2).contiguous()
        return mean.view(self._output_shape)

    @property
    def num_tasks(self):
        return self._output_shape[-1]

    def rsample(self, sample_shape=torch.Size(), base_samples=None, **kwargs):
        if base_samples is not None:
            # Make sure that the base samples agree with the distribution
            mean_shape = self.mean.shape
            base_sample_shape = base_samples.shape[-self.mean.ndimension() :]
            if mean_shape != base_sample_shape:
                raise RuntimeError(
                    "The shape of base_samples (minus sample shape dimensions) should agree with the shape "
                    "of self.mean. Expected ...{} but got {}".format(mean_shape, base_sample_shape)
                )
            sample_shape = base_samples.shape[: -self.mean.ndimension()]
            base_samples = base_samples.view(*sample_shape, *self.loc.shape)

        samples = super().rsample(sample_shape=sample_shape, base_samples=base_samples, **kwargs)
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = sample_shape + self._output_shape[:-2] + self._output_shape[:-3:-1]
            return samples.view(new_shape).transpose(-1, -2).contiguous()
        return samples.view(sample_shape + self._output_shape)

    def to_data_uncorrelated_dist(self, jitter_val=1e-4):
        """
        Convert a multitask QEP into a batched (non-multitask) QEPs
        The result retains the intertask covariances, but gets rid of the inter-data covariances.
        The resulting distribution will have ``len(qeps)`` tasks, and the tasks will be uncorrelated.

        :returns: the bached data-uncorrelated QEP
        :rtype: qpytorch.distributions.MultivariateQExponential
        """
        # Create batch distribution where all data are independent, but the tasks are dependent
        full_covar = self.lazy_covariance_matrix
        num_data, num_tasks = self.mean.shape[-2:]
        if self._interleaved:
            data_indices = torch.arange(0, num_data * num_tasks, num_tasks, device=full_covar.device).view(-1, 1, 1)
            task_indices = torch.arange(num_tasks, device=full_covar.device)
        else:
            data_indices = torch.arange(num_data, device=full_covar.device).view(-1, 1, 1)
            task_indices = torch.arange(0, num_data * num_tasks, num_data, device=full_covar.device)
        task_covars = full_covar[
            ..., data_indices + task_indices.unsqueeze(-2), data_indices + task_indices.unsqueeze(-1)
        ]
        return MultivariateQExponential(self.mean, to_linear_operator(task_covars).add_jitter(jitter_val=jitter_val), self.power)
    
    # to_data_independent_dist = to_data_uncorrelated_dist # alias to the same function with a more appropriate name

    @property
    def variance(self):
        var = super().variance
        if not self._interleaved:
            # flip shape of last two dimensions
            new_shape = self._output_shape[:-2] + self._output_shape[:-3:-1]
            return var.view(new_shape).transpose(-1, -2).contiguous()
        return var.view(self._output_shape)

    def __getitem__(self, idx) -> MultivariateQExponential:
        """
        Constructs a new MultivariateQExponential that represents a random variable
        modified by an indexing operation.

        The mean and covariance matrix arguments are indexed accordingly.

        :param Any idx: Index to apply to the mean. The covariance matrix is indexed accordingly.
        :returns: If indices specify a slice for samples and tasks, returns a
            MultitaskMultivariateQExponential, else returns a MultivariateQExponential.
        """

        # Normalize index to a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        if ... in idx:
            # Replace ellipsis '...' with explicit indices
            ellipsis_location = idx.index(...)
            if ... in idx[ellipsis_location + 1 :]:
                raise IndexError("Only one ellipsis '...' is supported!")
            prefix = idx[:ellipsis_location]
            suffix = idx[ellipsis_location + 1 :]
            infix_length = self.mean.dim() - len(prefix) - len(suffix)
            if infix_length < 0:
                raise IndexError(f"Index {idx} has too many dimensions")
            idx = prefix + (slice(None),) * infix_length + suffix
        elif len(idx) == self.mean.dim() - 1:
            # Normalize indices ignoring the task-index to include it
            idx = idx + (slice(None),)

        new_mean = self.mean[idx]

        # We now create a covariance matrix appropriate for new_mean
        if len(idx) <= self.mean.dim() - 2:
            # We are only indexing the batch dimensions in this case
            return MultitaskMultivariateQExponential(
                mean=new_mean,
                covariance_matrix=self.lazy_covariance_matrix[idx],
                power=self.power,
                interleaved=self._interleaved,
            )
        elif len(idx) > self.mean.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # We have an index that extends over all dimensions
            batch_idx = idx[:-2]
            if self._interleaved:
                row_idx = idx[-2]
                col_idx = idx[-1]
                num_rows = self._output_shape[-2]
                num_cols = self._output_shape[-1]
            else:
                row_idx = idx[-1]
                col_idx = idx[-2]
                num_rows = self._output_shape[-1]
                num_cols = self._output_shape[-2]

            if isinstance(row_idx, int) and isinstance(col_idx, int):
                # Single sample with single task
                row_idx = _normalize_index(row_idx, num_rows)
                col_idx = _normalize_index(col_idx, num_cols)
                new_cov = DiagLinearOperator(
                    self.lazy_covariance_matrix.diagonal()[batch_idx + (row_idx * num_cols + col_idx,)]
                )
                return MultivariateQExponential(mean=new_mean, covariance_matrix=new_cov, power=self.power)
            elif isinstance(row_idx, int) and isinstance(col_idx, slice):
                # A block of the covariance matrix
                row_idx = _normalize_index(row_idx, num_rows)
                col_idx = _normalize_slice(col_idx, num_cols)
                new_slice = slice(
                    col_idx.start + row_idx * num_cols,
                    col_idx.stop + row_idx * num_cols,
                    col_idx.step,
                )
                new_cov = self.lazy_covariance_matrix[batch_idx + (new_slice, new_slice)]
                return MultivariateQExponential(mean=new_mean, covariance_matrix=new_cov, power=self.power)
            elif isinstance(row_idx, slice) and isinstance(col_idx, int):
                # A block of the reversely interleaved covariance matrix
                row_idx = _normalize_slice(row_idx, num_rows)
                col_idx = _normalize_index(col_idx, num_cols)
                new_slice = slice(row_idx.start + col_idx, row_idx.stop * num_cols + col_idx, row_idx.step * num_cols)
                new_cov = self.lazy_covariance_matrix[batch_idx + (new_slice, new_slice)]
                return MultivariateQExponential(mean=new_mean, covariance_matrix=new_cov, power=self.power)
            elif (
                isinstance(row_idx, slice)
                and isinstance(col_idx, slice)
                and row_idx == col_idx == slice(None, None, None)
            ):
                new_cov = self.lazy_covariance_matrix[batch_idx]
                return MultitaskMultivariateQExponential(
                    mean=new_mean,
                    covariance_matrix=new_cov,
                    power=self.power,
                    interleaved=self._interleaved,
                    validate_args=False,
                )
            elif isinstance(row_idx, slice) or isinstance(col_idx, slice):
                # slice x slice or indices x slice or slice x indices
                if isinstance(row_idx, slice):
                    row_idx = torch.arange(num_rows)[row_idx]
                if isinstance(col_idx, slice):
                    col_idx = torch.arange(num_cols)[col_idx]
                row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing="ij")
                indices = (row_grid * num_cols + col_grid).reshape(-1)
                new_cov = self.lazy_covariance_matrix[batch_idx + (indices,)][..., indices]
                return MultitaskMultivariateQExponential(
                    mean=new_mean, covariance_matrix=new_cov, power=self.power, interleaved=self._interleaved, validate_args=False
                )
            else:
                # row_idx and col_idx have pairs of indices
                indices = row_idx * num_cols + col_idx
                new_cov = self.lazy_covariance_matrix[batch_idx + (indices,)][..., indices]
                return MultivariateQExponential(
                    mean=new_mean,
                    covariance_matrix=new_cov,
                    power=self.power
                )

    def __repr__(self) -> str:
        return f"MultitaskMultivariateQExponential(mean shape: {self._output_shape})"


def _normalize_index(i: int, dim_size: int) -> int:
    if i < 0:
        return dim_size + i
    else:
        return i


def _normalize_slice(s: slice, dim_size: int) -> slice:
    start = s.start
    if start is None:
        start = 0
    elif start < 0:
        start = dim_size + start
    stop = s.stop
    if stop is None:
        stop = dim_size
    elif stop < 0:
        stop = dim_size + stop
    step = s.step
    if step is None:
        step = 1
    return slice(start, stop, step)
