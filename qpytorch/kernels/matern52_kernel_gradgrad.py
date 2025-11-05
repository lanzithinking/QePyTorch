#!/usr/bin/env python3

import math

import torch
from linear_operator.operators import KroneckerProductLinearOperator

from gpytorch.kernels.matern_kernel import MaternKernel

sqrt5 = math.sqrt(5)
five_thirds = 5.0 / 3.0


class Matern52KernelGradGrad(MaternKernel):
    r"""
    Computes a covariance matrix of the Matern52 kernel that models the covariance
    between the values and first and second (non-mixed) partial derivatives for inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`.

    See :class:`qpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.Matern52KernelGradGrad())
        >>> covar = covar_module(x)  # Output: LinearOperator of size (110 x 110), where 110 = n * (2*d + 1)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.Matern52KernelGradGrad())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = qpytorch.kernels.ScaleKernel(qpytorch.kernels.Matern52KernelGradGrad(batch_shape=torch.Size([2]))) # noqa: E501
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 110 x 110)
    """

    def __init__(self, **kwargs):

        # remove nu in case it was set
        kwargs.pop("nu", None)
        super(Matern52KernelGradGrad, self).__init__(nu=2.5, **kwargs)
        self._interleaved = kwargs.pop('interleaved', True)

    def forward(self, x1, x2, diag=False, **params):

        lengthscale = self.lengthscale

        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        
        mask_idx1 = params.pop('mask_idx1', None) # mask off-diagonal covariance
        if mask_idx1 is not None:
            mask_idx2 = params.pop('mask_idx2', mask_idx1)
            assert mask_idx1.shape[:-1] == mask_idx2.shape[:-1], 'Batch shapes of mask indices do not match!'

        if not diag:

            K = torch.zeros(*batch_shape, n1 * (2 * d + 1), n2 * (2 * d + 1), device=x1.device, dtype=x1.dtype)

            distance_matrix = self.covar_dist(x1.div(lengthscale), x2.div(lengthscale), diag=diag, **params)
            exp_neg_sqrt5r = torch.exp(-sqrt5 * distance_matrix)
            one_plus_sqrt5r = 1 + sqrt5 * distance_matrix

            # differences matrix in each dimension to be used for derivatives
            # shape of n1 x n2 x d
            outer = x1.view(*batch_shape, n1, 1, d) - x2.view(*batch_shape, 1, n2, d)
            outer = outer / lengthscale.unsqueeze(-2) ** 2
            # shape of n1 x d x n2
            outer = torch.transpose(outer, -1, -2).contiguous()

            # 1) Kernel block, cov(f^m, f^n)
            # shape is n1 x n2
            # exp_component = torch.exp(-sqrt5 * distance_matrix)
            constant_component = one_plus_sqrt5r.add(five_thirds * distance_matrix**2)

            K[..., :n1, :n2] = constant_component * exp_neg_sqrt5r #exp_component

            # 2) First gradient block, cov(f^m, omega^n_i)
            outer1 = outer.view(*batch_shape, n1, n2 * d)
            # the - signs on -outer1 and -five_thirds cancel out
            K[..., :n1, n2: (n2 * (d + 1))] = five_thirds * outer1 * (one_plus_sqrt5r * exp_neg_sqrt5r).repeat(
                [*([1] * (n_batch_dims + 1)), d]
            )

            # 3) Second gradient block, cov(omega^m_j, f^n)
            outer2 = outer.transpose(-1, -3).reshape(*batch_shape, n2, n1 * d)
            outer2 = outer2.transpose(-1, -2)
            K[..., n1: (n1 * (d + 1)), :n2] = -five_thirds * outer2 * (one_plus_sqrt5r * exp_neg_sqrt5r).repeat(
                [*([1] * n_batch_dims), d, 1]
            )

            # 4) Hessian block, cov(omega^m_j, omega^n_i)
            outer3 = outer1.repeat([*([1] * n_batch_dims), d, 1]) * outer2.repeat([*([1] * (n_batch_dims + 1)), d])
            kp = KroneckerProductLinearOperator(
                torch.eye(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / lengthscale**2,
                torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
            )

            # part1 = -five_thirds * exp_neg_sqrt5r
            # part2 = 5 * outer3
            # part3 = 1 + sqrt5 * distance_matrix
            exp_neg_sqrt5rdd = exp_neg_sqrt5r.repeat([*([1] * (n_batch_dims)), d, d])

            K[..., n1: (n1 * (d + 1)), n2: (n2 * (d + 1))] = -five_thirds * exp_neg_sqrt5rdd.mul(
                # need to use kp.to_dense().mul instead of kp.to_dense().mul_
                # because otherwise a RuntimeError is raised due to how autograd works with
                # view + inplace operations in the case of 1-dimensional input
                (5 * outer3).sub(kp.to_dense().mul(one_plus_sqrt5r.repeat([*([1] * n_batch_dims), d, d])))
            )

            # 5) 1-3 block
            douter1dx2 = KroneckerProductLinearOperator(
                torch.ones(1, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2),
                torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
            ).to_dense()

            K_13 = five_thirds * (-douter1dx2 * one_plus_sqrt5r.repeat([*([1] * (n_batch_dims + 1)), d]) + 5* outer1 * outer1) * exp_neg_sqrt5r.repeat(
                [*([1] * (n_batch_dims + 1)), d]
            )  # verified for n1=n2=1 case
            K[..., :n1, (n2 * (d + 1)) :] = K_13

            if d>1:
                douter1dx2 = KroneckerProductLinearOperator(
                    (torch.ones(1, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2)).transpose(-1, -2),
                    torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
                ).to_dense()
            K_31 = five_thirds * (-douter1dx2 * one_plus_sqrt5r.repeat([*([1] * n_batch_dims), d, 1]) + 5* outer2 * outer2) * exp_neg_sqrt5r.repeat(
                [*([1] * n_batch_dims), d, 1]
            )  # verified for n1=n2=1 case
            K[..., (n1 * (d + 1)) :, :n2] = K_31

            # rest of the blocks are all of size (n1*d,n2*d)
            outer1 = outer1.repeat([*([1] * n_batch_dims), d, 1])
            outer2 = outer2.repeat([*([1] * (n_batch_dims + 1)), d])
            # II = (torch.eye(d,d,device=x1.device,dtype=x1.dtype)/lengthscale.pow(2)).repeat(*batch_shape,n1,n2)
            kp2 = KroneckerProductLinearOperator(
                torch.ones(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2),
                torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
            ).to_dense()

            # II may not be the correct thing to use. It might be more appropriate to use kp instead??
            II = kp.to_dense()
            # exp_neg_sqrt5rdd = exp_neg_sqrt5r.repeat([*([1] * (n_batch_dims)), d, d])
            invrdd = (distance_matrix+self.eps).pow(-1)
            # invrdd[torch.arange(min(n1,n2)),torch.arange(min(n1,n2))] = distance_matrix.diagonal()
            invrdd = invrdd.repeat([*([1] * (n_batch_dims)), d, d])
            # invrdd = distance_matrix.pow(-1).fill_diagonal_(0).repeat([*([1] * (n_batch_dims)), d, d]).fill_diagonal_(1)

            K_23 = five_thirds * 5* ((kp2 - sqrt5*invrdd* outer1 * outer1) * outer2 + 2.0 * II * outer1) * exp_neg_sqrt5rdd  # verified for n1=n2=1 case

            K[..., n1 : (n1 * (d + 1)), (n2 * (d + 1)) :] = K_23

            if d>1:
                kp2t = KroneckerProductLinearOperator(
                    (torch.ones(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.pow(2)).transpose(-1, -2),
                    torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1),
                ).to_dense()
            K_32 = five_thirds * 5* (
                (-(kp2t if d>1 else kp2) + sqrt5*invrdd* outer2 * outer2) * outer1 - 2.0 * II * outer2
            ) * exp_neg_sqrt5rdd  # verified for n1=n2=1 case

            K[..., (n1 * (d + 1)) :, n2 : (n2 * (d + 1))] = K_32

            # K_33 = five_thirds * 5*(
            #     ((-(kp2t if d>1 else kp2) + sqrt5*invrdd*outer2 * outer2) * (-kp2) - 2.0 *sqrt5*invrdd * II * outer2 * outer1 + 2.0 * (II) ** 2
            # )  + (
            #     (-(kp2t if d>1 else kp2)*sqrt5*invrdd + (5+sqrt5*invrdd)*invrdd**2*outer2 * outer2) * outer1 - 2.0 *sqrt5*invrdd * II * outer2
            # ) * outer1) * exp_neg_sqrt5rdd  # verified for n1=n2=1 case
            K_33 = five_thirds * 5*(
                (kp2 - sqrt5*invrdd*outer1 * outer1) * ((kp2t if d>1 else kp2)-sqrt5*invrdd* outer2 * outer2) 
                + sqrt5*invrdd*(invrdd**2*outer3-4*(II))*outer3 + 2*(II)**2
            ) * exp_neg_sqrt5rdd

            K[..., (n1 * (d + 1)) :, (n2 * (d + 1)) :] = K_33

            # Symmetrize for stability
            if n1 == n2 and torch.eq(x1, x2).all():
                K = 0.5 * (K.transpose(-1, -2) + K)

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            if self._interleaved:
                pi1 = torch.arange(n1 * (2 * d + 1)).view(2 * d + 1, n1).t().reshape((n1 * (2 * d + 1)))
                pi2 = torch.arange(n2 * (2 * d + 1)).view(2 * d + 1, n2).t().reshape((n2 * (2 * d + 1)))
                K = K[..., pi1, :][..., :, pi2]

            if mask_idx1 is not None:
                diag2keep = K.diagonal(dim1=-1, dim2=-2)[mask_idx1&mask_idx2]
                keep_idx = (~mask_idx1).unsqueeze(-1) & (~mask_idx2).unsqueeze(-2)
                K = K * keep_idx
                K.diagonal(dim1=-1, dim2=-2)[mask_idx1&mask_idx2] = diag2keep * self.eps
                # if mask_idx1.ndim==1:
                #     diag2keep = K[...,mask_idx1,mask_idx2]
                #     K[...,mask_idx1,:] = 0; K[...,mask_idx2] = 0
                #     K[...,mask_idx1,mask_idx2] = diag2keep * self.eps
                # elif mask_idx1.ndim==2:
                #     for b in range(mask_idx1.shape[0]):
                #         diag2keep = K[b,...,mask_idx1[b],mask_idx2[b]]
                #         K[b,...,mask_idx1[b],:] = 0; K[b,...,mask_idx2[b]] = 0
                #         K[b,...,mask_idx1[b],mask_idx2[b]] = diag2keep * self.eps
                # else:
                #     raise NotImplementedError('Mask indices of batch dimension bigger than 1 not implemented!')

            return K
        else:
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError("diag=True only works when x1 == x2")

            # nu is set to 2.5
            kernel_diag = super(Matern52KernelGradGrad, self).forward(x1, x2, diag=True)
            grad_diag = (
                five_thirds * torch.ones(*batch_shape, n2, d, device=x1.device, dtype=x1.dtype)
            ) / lengthscale**2
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * d)
            gradgrad_diag = (
                5**2 * torch.ones(*batch_shape, n2, d, device=x1.device, dtype=x1.dtype) / lengthscale.pow(4)
            )
            gradgrad_diag = gradgrad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * d)
            k_diag = torch.cat((kernel_diag, grad_diag, gradgrad_diag), dim=-1)
            if self._interleaved:
                pi = torch.arange(n2 * (2 * d + 1)).view(2 * d + 1, n2).t().reshape((n2 * (2 * d + 1)))
                k_diag = k_diag[..., pi]
            return k_diag

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) * 2 + 1
