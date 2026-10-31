"""
Exact computation of inter-layer input Hessian blocks H^f_{v,w}.

H^f_{v,w} = d^2L / df_v df_w  where f_v, f_w are intermediate activations.

Strategy (sequential models with N-segment split):

  For idx_v <= idx_w:
    1. cached = forward segments[0..idx_v] without graph -> f_v_cached.
    2. f_v = f_v_cached.detach().requires_grad_(True) - leaf.
    3. Forward with graph: f_v -> segments[idx_v+1..idx_w] -> f_w.
    4. Forward with graph: f_w -> segments[idx_w+1..N] -> logits -> loss.
    5. grad_w = dL/df_w  (via autograd, create_graph=True).
    6. H_{v,w}[i, j] = d(grad_w[j]) / d(f_v[i]).

  For the diagonal block idx_v == idx_w:
    H_{v,v} = d^2L/df_v^2 - standard Hessian w.r.t. f_v.

The model implements the SegmentedModel protocol:
  - get_layer_names() -> N names -> measurement points.
  - get_segments() -> N+1 callable (seg[0]: x->f_0,..., seg[N]: f_{N-1}->logits).

hessian_of_loss_wrt_tensor and cross_hessian take only the loss and the two
activation tensors, so a model whose forward graph is not a chain - and so cannot
satisfy SegmentedModel - computes its blocks by calling them directly on its own
tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from .types import BlockHessianResult, HessianBlock, LayerID


class SegmentedModel(Protocol):
    """A model that can be split into segments for block-wise Hessian computation."""

    def get_layer_names(self) -> list[str]: ...
    def get_segments(self) -> list[Callable[[Tensor], Tensor]]: ...


def hessian_of_loss_wrt_tensor(loss: Tensor, fv: Tensor) -> Tensor:
    """d^2L/df_v^2 - full Hessian matrix (d_v x d_v), averaged over the batch.

    H_avg = (1/B) sum_b d^2L_b / df_v^2.

    loss = (1/B) sum_b L_b  (reduction='mean'), so
    grad_fv[b] = (1/B) dL_b/df_v.  col_sum = sum_b grad[b,j] = (1/B) sum_b dL_b/df_v[j].
    row = d(col_sum)/d(fv):  row[b',i] = (1/B) d^2L_{b'}/df_v[i]df_v[j].
    Sum over batch -> (1/B) sum_b H_b = H_avg.  (Not averaged again!)
    """
    B = fv.shape[0]
    flat_dim = fv[0].numel()

    grad_fv = torch.autograd.grad(loss, fv, create_graph=True)[0]
    grad_flat = grad_fv.reshape(B, flat_dim)

    H = torch.zeros(flat_dim, flat_dim, device=fv.device)
    for j in range(flat_dim):
        col_sum = grad_flat[:, j].sum()
        row = torch.autograd.grad(col_sum, fv, retain_graph=True)[0]
        H[j, :] = row.reshape(B, flat_dim).sum(0)

    return H


def cross_hessian(loss: Tensor, fv: Tensor, fw: Tensor) -> Tensor:
    """d^2L / df_v df_w - cross block (d_v x d_w), averaged over the batch.

    H_avg[i,j] = (1/B) sum_b d^2L_b / df_v[i] df_w[j].

    loss = (1/B) sum_b L_b  ->  grad_fw[b,j] = (1/B) dL_b/df_w[j].
    col_sum = sum_b grad_fw[b,j] = (1/B) sum_b dL_b/df_w[j].
    row = d(col_sum)/d(fv):  row[b',i] = (1/B) d^2L_{b'}/df_v[i]df_w[j].
    Sum over batch -> (1/B) sum_b H_b = H_avg.
    """
    B = fv.shape[0]
    d_v = fv[0].numel()
    d_w = fw[0].numel()

    grad_fw = torch.autograd.grad(loss, fw, create_graph=True)[0]
    grad_fw_flat = grad_fw.reshape(B, d_w)

    H = torch.zeros(d_v, d_w, device=fv.device)
    for j in range(d_w):
        col_sum = grad_fw_flat[:, j].sum()
        row = torch.autograd.grad(col_sum, fv, retain_graph=True)[0]
        H[:, j] = row.reshape(B, d_v).sum(0)

    return H


class ExactBlockHessian:
    """Exact block-wise computation of the input Hessian H^f_{v,w}."""

    def __init__(
        self,
        model: SegmentedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_block(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> HessianBlock:
        """Computes a single block H^f_{v,w} = d^2L / df_v df_w."""
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        swapped = False
        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v
            layer_v, layer_w = layer_w, layer_v
            swapped = True

        # 1. Forward to f_v without graph
        f_v_cached = self._forward_no_grad(x, segments, up_to=idx_v)
        f_v = f_v_cached.detach().requires_grad_(True)

        # 2. Forward f_v -> f_w -> loss - single graph
        with torch.enable_grad():
            f_w = f_v
            for i in range(idx_v + 1, idx_w + 1):
                f_w = segments[i](f_w)
            h = f_w
            for i in range(idx_w + 1, len(segments)):
                h = segments[i](h)
            loss = self._loss_fn(h, y)

        if idx_v == idx_w:
            # Diagonal block: H_{v,v} = d^2L/df_v^2
            block_matrix = hessian_of_loss_wrt_tensor(loss, f_v)
        else:
            # Cross block: H_{v,w}[i,j] = d^2L / df_v[i] df_w[j]
            block_matrix = cross_hessian(loss, f_v, f_w)

        if swapped:
            block_matrix = block_matrix.t()

        return HessianBlock(
            layer_v=layer_v,
            layer_w=layer_w,
            matrix=block_matrix.detach(),
        )

    def compute_all_blocks(
        self,
        x: Tensor,
        y: Tensor,
        pairs: list[tuple[LayerID, LayerID]] | None = None,
    ) -> BlockHessianResult:
        """Computes blocks for the specified pairs (or all pairs of layer_names)."""
        names = self._model.get_layer_names()
        if pairs is None:
            pairs = [(v, w) for v in names for w in names]

        blocks: dict[tuple[LayerID, LayerID], HessianBlock] = {}
        for v, w in pairs:
            blocks[(v, w)] = self.compute_block(x, y, v, w)

        # layer_dims
        segments = self._model.get_segments()
        layer_dims: dict[LayerID, int] = {}
        h = x
        for i, seg in enumerate(segments):
            with torch.no_grad():
                h = seg(h)
            if i < len(names):
                layer_dims[names[i]] = h.numel() // h.shape[0]

        return BlockHessianResult(blocks=blocks, layer_dims=layer_dims)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_no_grad(
        x: Tensor,
        segments: list[Callable[[Tensor], Tensor]],
        up_to: int,
    ) -> Tensor:
        h = x
        with torch.no_grad():
            for i in range(up_to + 1):
                h = segments[i](h)
        return h
