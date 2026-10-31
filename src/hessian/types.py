from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

LayerID = str


@dataclass
class HessianBlock:
    """A single block H^f_{v,w} of the inter-layer Hessian."""

    layer_v: LayerID
    layer_w: LayerID
    matrix: Tensor  # shape (d_v, d_w)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.matrix.shape)  # type: ignore[return-value]

    @property
    def frobenius_norm(self) -> float:
        return torch.linalg.norm(self.matrix, ord="fro").item()

    @property
    def spectral_norm(self) -> float:
        return torch.linalg.norm(self.matrix, ord=2).item()

    @property
    def stable_rank(self) -> float:
        """||H||_F^2 / ||H||_2^2 - effective dimensionality of the block.

        For a non-zero matrix always D in [1, rank(H)].
        For a zero matrix (||H||_2 < eps) returns NaN (D is undefined).
        """
        sn = self.spectral_norm
        if sn < 1e-12:
            return math.nan
        fn = self.frobenius_norm
        return (fn / sn) ** 2


@dataclass
class GNGapResult:
    """Result of the H = H^GN + H^T decomposition for a single block (v,w)."""

    gap: float  # ||H^T||_F / (||H^GN||_F + eps)
    frob_full: float  # ||H^f||_F
    frob_gn: float  # ||H^GN||_F
    frob_tensor: float  # ||H^T||_F


@dataclass
class BlockHessianResult:
    """Collection of inter-layer Hessian blocks for all layer pairs."""

    blocks: dict[tuple[LayerID, LayerID], HessianBlock]
    layer_dims: dict[LayerID, int]

    def get_block(self, v: LayerID, w: LayerID) -> HessianBlock | None:
        return self.blocks.get((v, w))
