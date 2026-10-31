"""Decomposition H = H^GN + H^T and computation of GN-Gap.

GN block for CE-loss:
  H^GN_{v,w} = (1/B) sum_b  J_v[b]^T  L''[b]  J_w[b]

where J_v[b] = d logits[b] / d f_v[b]  in R^{d_out x d_v},
      L''[b] = diag(p[b]) - p[b] p[b]^T  in R^{d_out x d_out}  (Hessian of CE w.r.t. logits).

Tensor component:
  H^T_{v,w} = H^f_{v,w} - H^GN_{v,w}

GN-Gap:
  Gap(v,w) = ||H^T_{v,w}||_F / (||H^GN_{v,w}||_F + eps)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ._estimators import rademacher_signs
from .exact import ExactBlockHessian, SegmentedModel
from .types import GNGapResult, HessianBlock, LayerID

_EPS = 1e-12


def _assert_mean_reduction(loss_fn: Callable[[Tensor, Tensor], Tensor]) -> None:
    """Asserts reduction='mean' - all normalization is tied to this."""
    reduction = getattr(loss_fn, "reduction", "mean")
    if reduction != "mean":
        raise ValueError(
            f"GN decomposition assumes reduction='mean', got reduction='{reduction}'"
        )


# ======================================================================
# Exact
# ======================================================================


class ExactGNDecomposition:
    """Exact computation of H^GN_{v,w} and GN-Gap via Jacobians."""

    def __init__(
        self,
        model: SegmentedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        _assert_mean_reduction(loss_fn)
        self._model = model
        self._loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_gn_block(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> HessianBlock:
        """H^GN_{v,w} = (1/B) sum_b J_v^T L'' J_w."""
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        swapped = False
        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v
            layer_v, layer_w = layer_w, layer_v
            swapped = True

        jac_v, jac_w, logits = self._compute_jacobians(
            x,
            segments,
            idx_v,
            idx_w,
        )
        gn_matrix = self._assemble_gn(jac_v, jac_w, logits)

        if swapped:
            gn_matrix = gn_matrix.t()

        return HessianBlock(layer_v=layer_v, layer_w=layer_w, matrix=gn_matrix.detach())

    def compute_gn_gap(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
        full_block: HessianBlock | None = None,
    ) -> GNGapResult:
        """GN-Gap = ||H^T||_F / (||H^GN||_F + eps) for the pair (v,w).

        Args:
            full_block: precomputed block H^f_{v,w}. If None, computed automatically.
        """
        gn_block = self.compute_gn_block(x, y, layer_v, layer_w)

        if full_block is None:
            hess = ExactBlockHessian(self._model, self._loss_fn)
            full_block = hess.compute_block(x, y, layer_v, layer_w)

        h_t = full_block.matrix - gn_block.matrix
        frob_full = full_block.frobenius_norm
        frob_gn = gn_block.frobenius_norm
        frob_tensor = torch.linalg.norm(h_t, ord="fro").item()

        return GNGapResult(
            gap=frob_tensor / (frob_gn + _EPS),
            frob_full=frob_full,
            frob_gn=frob_gn,
            frob_tensor=frob_tensor,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_jacobians(
        self,
        x: Tensor,
        segments: list[Callable[[Tensor], Tensor]],
        idx_v: int,
        idx_w: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes J_v, J_w and logits in a single forward graph pass.

        Returns:
            jac_v: (B, d_out, d_v) - d logits / d f_v
            jac_w: (B, d_out, d_w) - d logits / d f_w
            logits: (B, d_out)
        """
        # Forward to f_v without graph
        f_v = ExactBlockHessian._forward_no_grad(x, segments, up_to=idx_v)
        f_v = f_v.detach().requires_grad_(True)

        # Forward f_v -> f_w -> logits with graph
        h = f_v
        for i in range(idx_v + 1, idx_w + 1):
            h = segments[i](h)
        f_w = h
        for i in range(idx_w + 1, len(segments)):
            h = segments[i](h)
        logits = h  # (B, d_out)

        B, d_out = logits.shape
        d_v = f_v.shape[1]

        # J_v: (B, d_out, d_v)
        jac_v = self._batch_jacobian(logits, f_v, B, d_out, d_v)

        if idx_v == idx_w:
            jac_w = jac_v
        else:
            d_w = f_w.shape[1]
            jac_w = self._batch_jacobian(logits, f_w, B, d_out, d_w)

        return jac_v, jac_w, logits.detach()

    @staticmethod
    def _batch_jacobian(
        logits: Tensor,
        f: Tensor,
        B: int,
        d_out: int,
        d_f: int,
    ) -> Tensor:
        """d logits/d f -> (B, d_out, d_f) via d_out backward passes."""
        jac = torch.zeros(B, d_out, d_f, device=f.device)
        for k in range(d_out):
            grad_k = torch.autograd.grad(
                logits[:, k].sum(),
                f,
                retain_graph=True,
            )[0]  # (B, d_f)
            jac[:, k, :] = grad_k
        return jac

    @staticmethod
    def _assemble_gn(
        jac_v: Tensor,
        jac_w: Tensor,
        logits: Tensor,
    ) -> Tensor:
        """H^GN = (1/B) sum_b J_v^T L'' J_w.

        L''[b] = diag(p) - pp^T  (Hessian of CE loss w.r.t. logits).
        """
        B = jac_v.shape[0]
        p = F.softmax(logits, dim=-1)  # (B, d_out)
        # L'' @ J_w: (B, d_out, d_w)
        # (diag(p) - pp^T) @ J_w = p.unsqueeze(-1) * J_w - p @ (p^T @ J_w)
        pJ_w = p.unsqueeze(-1) * jac_w  # (B, d_out, d_w)
        pTJ_w = torch.bmm(p.unsqueeze(1), jac_w)  # (B, 1, d_w)
        L2_Jw = pJ_w - p.unsqueeze(-1) * pTJ_w  # (B, d_out, d_w)
        # J_v^T @ L'' @ J_w -> (B, d_v, d_w)
        gn_per_sample = torch.bmm(jac_v.transpose(1, 2), L2_Jw)
        # 1/B normalization: gn_per_sample.sum(0) = sum_b J_v^T L'' J_w,
        # loss = (1/B) sum L_b => H_avg = (1/B) sum H_b => GN_avg = (1/B) sum GN_b.
        return gn_per_sample.sum(0) / B  # (d_v, d_w)


# ======================================================================
# Stochastic
# ======================================================================


class StochasticGNGapEstimator:
    """Stochastic estimation of ||H^T||_F and GN-Gap via Hutchinson.

    For each Rademacher probe z:
      1. H^f z - standard HVP.
      2. H^GN z = J_v^T L'' (J_w z) - analytic GN-HVP.
      3. H^T z = H^f z - H^GN z  (exact difference, no extra variance).
    """

    def __init__(
        self,
        model: SegmentedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_probes: int = 30,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        _assert_mean_reduction(loss_fn)
        self._model = model
        self._loss_fn = loss_fn
        self._n_probes = n_probes
        self._generator = generator

    def estimate_gn_gap(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> GNGapResult:
        """Estimates GN-Gap via Hutchinson with shared probe vectors."""
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v

        # Forward to f_v without graph
        f_v = ExactBlockHessian._forward_no_grad(x, segments, up_to=idx_v)
        f_v = f_v.detach().requires_grad_(True)

        # Forward f_v -> f_w -> logits -> loss with graph
        h = f_v
        for i in range(idx_v + 1, idx_w + 1):
            h = segments[i](h)
        f_w = h

        h_cont = f_w
        for i in range(idx_w + 1, len(segments)):
            h_cont = segments[i](h_cont)
        logits = h_cont  # (B, d_out)
        loss = self._loss_fn(logits, y)

        # dL/df_w (for full HVP)
        grad_fw = torch.autograd.grad(loss, f_w, create_graph=True, retain_graph=True)[
            0
        ]

        # J_w: (B, d_out, d_w) - for GN-HVP
        B, d_out = logits.shape
        d_w = f_w[0].numel()  # C*H*W for 4D, d for 2D
        jac_w = self._batch_jacobian_retained(logits, f_w, B, d_out, d_w)

        # p = softmax(logits) - for L''
        p = F.softmax(logits.detach(), dim=-1)  # (B, d_out)

        device = f_v.device

        # Not `hutchinson_frob_sq`: the same probe must feed the full, GN and tensor
        # closures each iteration, so the three estimates carry one shared probe's
        # noise and their difference is meaningful; the shared kernel draws its own
        # probe per call, so three calls to it would not share one.
        sq_full, sq_gn, sq_tensor = 0.0, 0.0, 0.0
        for _ in range(self._n_probes):
            # Shared probe: same z for all samples (as in StochasticHessianEstimator)
            z_shared = rademacher_signs((d_w,), device, self._generator)
            # z_flat (B, d_w) - for bmm with jac_w; z_shaped (B, *spatial) - for dot with grad_fw
            z_flat = z_shared.unsqueeze(0).expand(B, -1)
            z_shaped = z_shared.reshape(1, *f_w.shape[1:]).expand_as(f_w)

            # 1. Full HVP: H^f_{v,w} z
            dot = (grad_fw * z_shaped).sum()
            hvp_full = torch.autograd.grad(dot, f_v, retain_graph=True)[0]
            # H_avg z = sum over batch (each hvp_full[b] = (1/B) * H_b z)
            Hz_full = hvp_full.sum(0)

            # 2. GN-HVP: J_v^T L'' (J_w z)
            gn_hvp = self._gn_hvp(
                logits,
                f_v,
                jac_w,
                p,
                z_flat,
                B,
            )
            Hz_gn = gn_hvp.sum(0)

            # 3. Tensor HVP
            Hz_tensor = Hz_full - Hz_gn

            sq_full += (Hz_full**2).sum().item()
            sq_gn += (Hz_gn**2).sum().item()
            sq_tensor += (Hz_tensor**2).sum().item()

        frob_full = (sq_full / self._n_probes) ** 0.5
        frob_gn = (sq_gn / self._n_probes) ** 0.5
        frob_tensor = (sq_tensor / self._n_probes) ** 0.5

        return GNGapResult(
            gap=frob_tensor / (frob_gn + _EPS),
            frob_full=frob_full,
            frob_gn=frob_gn,
            frob_tensor=frob_tensor,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_jacobian_retained(
        logits: Tensor,
        f: Tensor,
        B: int,
        d_out: int,
        d_f: int,
    ) -> Tensor:
        """d logits/d f -> (B, d_out, d_f), retain_graph=True."""
        jac = torch.zeros(B, d_out, d_f, device=f.device)
        for k in range(d_out):
            grad_k = torch.autograd.grad(
                logits[:, k].sum(),
                f,
                retain_graph=True,
            )[0]
            jac[:, k, :] = grad_k.reshape(B, -1)
        return jac.detach()

    @staticmethod
    def _gn_hvp(
        logits: Tensor,
        f_v: Tensor,
        jac_w: Tensor,
        p: Tensor,
        z: Tensor,
        B: int,
    ) -> Tensor:
        """H^GN_{v,w} z = J_v^T L'' (J_w z).

        Computes J_w z -> L''(J_w z) analytically -> VJP through logits -> d/d f_v.
        """
        # J_w z -> (B, d_out)
        jw_z = torch.bmm(jac_w, z.unsqueeze(-1)).squeeze(-1)  # (B, d_out)

        # L'' (J_w z) = diag(p)(J_w z) - p(p^T (J_w z))
        # = p * jw_z - p * (p * jw_z).sum(-1, keepdim=True)
        pT_jw_z = (p * jw_z).sum(-1, keepdim=True)  # (B, 1)
        u = p * jw_z - p * pT_jw_z  # (B, d_out)

        # VJP: J_v^T u = d(logits * u.detach()).sum() / d f_v
        # Divide by B to match hvp_full (mean-reduced loss).
        u_detached = u.detach()
        scalar = (logits * u_detached).sum() / B
        gn_hvp = torch.autograd.grad(scalar, f_v, retain_graph=True)[0]

        return gn_hvp
