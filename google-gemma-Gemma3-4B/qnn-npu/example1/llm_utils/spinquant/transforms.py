import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ._hadamard_utils import hadamard_transform


class Transformation(nn.Module):
    """
    Abstract class for transformations.
    """

    def __init__(self, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

    def forward(self, x, inv_t=False):
        if inv_t:
            return self.apply_inverse(x)
        return self.apply_transform(x)

    def apply_transform(self, x: Tensor, **kwargs) -> Tensor:
        if self.use_checkpointing:
            # we always checkpoint this, because they are usually cheap yet take up valuable space
            return checkpoint(self._apply_transform, x, use_reentrant=False, **kwargs)
        else:
            return self._apply_transform(x, **kwargs)

    def apply_inverse(self, x: Tensor, **kwargs) -> Tensor:
        # we always checkpoint this, because they are usually cheap yet take up valuable space
        if self.use_checkpointing:
            return checkpoint(self._apply_inverse, x, use_reentrant=False, **kwargs)
        else:
            return self._apply_inverse(x, **kwargs)

    def merge_apply_after_linear(self, W: Tensor, bias: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Description:
        Merges the forward transformation on the right of the weight, i.e. outputs (W.T @ T).T

        Details:
        Instead of applying (X @ W.T) @ T, we can merge X @ (W.T @ T).
        The latter is more efficient when this can be done once (e.g. when W and T are fixed)

        Note that when there is a bias term, we do need to fix this too.
        """
        if bias is not None:
            bias = self.apply_transform(bias)
        return self.apply_transform(W.T).T, bias

    def merge_inverse_before_linear(self, W: Tensor) -> Tensor:
        """
        Description:
        Merges the inverse transformation ont he left of the weight, i.e. outputs T^(-1) @ W

        Details:
        Instead of applying (X @ T^-1) @ W, we can merge X @ (T^-1 @ W) = X @ (W.T @ T^-T).T.
        The latter is more efficient when this can be done once (e.g. when W and T are fixed) and can also
        improve quantization of the weights.
        N.B., because linear weights in pytorch are defined as (out_features, in_features), we do not need
        to transpose the input and output
        """
        return self.apply_inverse(W, transpose=True)  # identical to self.apply_inverse(torch.eye(self.size)) @ W

    def _apply_transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _apply_inverse(self, x: Tensor, transpose: bool = False) -> Tensor:
        """
        Applies the inverse to a tensor.

        Note that we only consider linear transformations that could
        be described as a matrix multiplication X @ M^-1. This is useful for merging weight, see self.merge_inverse_before_linear
        Argument transform=True should output the tranposed inverse, i.e. X @ M^(-T). In many cases we can avoid the explicit computation of M^(-T).
        Tip: for development, one can use the lazy implementation:
        if transpose:
            return x @ self.apply_inverse(torch.eye(x.shape[-1]).to(x)).T
        though this is slow and not reccommended
        """
        raise NotImplementedError

    def to_eval_mode(self):
        # This can be overwritten for inference speed up, see KroneckerTransform.
        pass


class HadamardTransform(Transformation):
    def __init__(
        self,
        size,
        init_type="randomized_hadamard",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.size = size
        if init_type == "randomized_hadamard":
            self.s = nn.Parameter(
                torch.bernoulli(0.5 * torch.ones(size, device=device)) * 2 - 1,
            )
        else:
            self.s = nn.Parameter(torch.ones(size, device=device))
        self.scale = 1 / math.sqrt(2 ** math.ceil(math.log2(size)))

    def _apply_transform(self, x: Tensor, inverse_s: bool = False) -> Tensor:
        """
        Applies the Hadamard transform to the input tensor x.
        Equivalent to X @ H, where H is a (randomized-row) Hadamard matrix.
        Args:
            x (Tensor): Input tensor of shape (..., size).
        Returns:
            Tensor: Output tensor of shape (..., size).
        """

        assert x.shape[-1] == self.size, f"Size of input {x.shape[-1]} does not match size of transform {self.size}"
        out = hadamard_transform(self.s * x, scale=self.scale, handle_no_power_2="factor")

        return out

    def _apply_inverse(self, x: Tensor, transpose: bool = True) -> Tensor:
        """
        Applies the inverse of the Hadamard transform to the input tensor x.
        Equivalent to X @ H^-1 = X @ H^T, where H is a (randomized-row) Hadamard matrix.
        Args:
            x (Tensor): Input tensor of shape (..., size).
        Returns:
            Tensor: Output tensor of shape (..., size).
        """
        if transpose:
            return (hadamard_transform(x.T / self.s, scale=self.scale, handle_no_power_2="factor")).T
