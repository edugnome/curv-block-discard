"""
Stochastic estimation of ||H^f_{v,w}||_F via Hutchinson + HVP.

Frobenius norm estimation without building the full matrix:
  ||H||_F^2 = tr(H^T H) ~= (1/m) sum_k z_k^T H^T H z_k = (1/m) sum_k ||H z_k||^2

where z_k are random Rademacher vectors of dimension d_w.

HVP H_{v,w} z is computed via double backward:
  H_{v,w} z = d/df_v [ (dL/df_w)^T z ]

Stable rank D = ||H||_F^2 / ||H||_2^2 estimation via power iteration:
  ||H||_2 is computed by the iteration v <- H^T(Hv) / ||H^T(Hv)|| (converges to sigma_1).
  Adjoint HVP: H^T u = d/df_w [ (dL/df_v)^T u ].

Complexity: O(m * P) in time, O(P) in memory (m is the number of probes).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ._estimators import HVPClosure, hutchinson_frob_sq, power_iteration_spectral_sq
from .exact import SegmentedModel
from .metrics import coupling_from_norms
from .types import LayerID


class StochasticHessianEstimator:
    """Estimation of ||H^f_{v,w}||_F and stable rank via Hutchinson / HVP.

    ``generator`` (optional) drives every Hutchinson draw - the shared probe vectors and
    the power-iteration start vector - so a seeded generator fixes the estimates; it must
    live on the activations' device. ``None`` uses that device's global RNG
    (``torch.manual_seed`` then fixes the estimates).
    """

    def __init__(
        self,
        model: SegmentedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_probes: int = 30,
        n_power_iter: int = 20,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._n_probes = n_probes
        self._n_power_iter = n_power_iter
        self._generator = generator

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graph(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
        *,
        need_grad_fv: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Builds the forward graph for HVP computation.

        Returns:
            (f_v, f_w_graph, grad_fw, grad_fv | None)
        """
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v

        h = x
        with torch.no_grad():
            for i in range(idx_v + 1):
                h = segments[i](h)
        f_v = h.detach().requires_grad_(True)

        h = f_v
        for i in range(idx_v + 1, idx_w + 1):
            h = segments[i](h)
        f_w_graph = h

        h = f_w_graph
        for i in range(idx_w + 1, len(segments)):
            h = segments[i](h)
        loss = self._loss_fn(h, y)

        grad_fw = torch.autograd.grad(
            loss,
            f_w_graph,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_fv: Tensor | None = None
        if need_grad_fv:
            grad_fv = torch.autograd.grad(
                loss,
                f_v,
                create_graph=True,
                retain_graph=True,
            )[0]

        return f_v, f_w_graph, grad_fw, grad_fv

    # ------------------------------------------------------------------
    # Frobenius norm
    # ------------------------------------------------------------------

    def estimate_block_frob_norm(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """Estimates ||H_avg^f_{v,w}||_F via Hutchinson with a shared probe vector.

        ||H_avg||_F^2 = E_z[||H_avg z||^2],  z ~ Rademacher(d_w).
        H_avg z is computed via HVP: expand z to the batch dimension, then average the result.
        """
        f_v, f_w_graph, grad_fw, _ = self._setup_graph(x, y, layer_v, layer_w)

        device = f_v.device
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        frob_sq = self._hutchinson_shared(f_v, f_w_graph, grad_fw, d_w, device)
        return frob_sq**0.5

    # ------------------------------------------------------------------
    # Stable rank (dimension metric)
    # ------------------------------------------------------------------

    def estimate_stable_rank(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """D(v,w) = ||H||_F^2 / ||H||_2^2 via Hutchinson + power iteration.

        Both norms are computed on the batch-averaged Hessian H_avg
        using shared probe vectors.
        """
        f_v, f_w_graph, grad_fw, grad_fv = self._setup_graph(
            x,
            y,
            layer_v,
            layer_w,
            need_grad_fv=True,
        )

        device = f_v.device
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        # --- ||H||_2 via power iteration ---
        spectral_sq = self._power_iteration(
            f_v,
            f_w_graph,
            grad_fw,
            grad_fv,
            d_w,
            device,
        )

        if spectral_sq < 1e-24:
            return math.nan
        frob_sq = self._hutchinson_shared(
            f_v,
            f_w_graph,
            grad_fw,
            d_w,
            device,
        )

        return frob_sq / spectral_sq

    @staticmethod
    def _hvp_closure(
        grad_src: Tensor,
        src_graph: Tensor,
        target: Tensor,
    ) -> HVPClosure:
        """Builds the batch-averaged HVP closure z |-> H_avg z.

        Expands the probe across the batch, contracts with grad_src = dL/d(src_graph),
        differentiates w.r.t. ``target``, and sums over the batch:
          (1/B) sum_b H_b z = H_avg z   (loss uses reduction='mean').

        With (grad_src, src_graph, target) = (grad_fw, f_w, f_v) this is H_avg z;
        with (grad_fv, f_v, f_w) it is the adjoint H_avg^T u.
        """

        def hvp(z: Tensor) -> Tensor:
            z_batch = z.reshape(1, *src_graph.shape[1:]).expand_as(src_graph)
            dot = (grad_src * z_batch).sum()
            return torch.autograd.grad(dot, target, retain_graph=True)[0].sum(0)

        return hvp

    def _power_iteration(
        self,
        f_v: Tensor,
        f_w_graph: Tensor,
        grad_fw: Tensor,
        grad_fv: Tensor | None,
        d_w: int,
        device: torch.device,
    ) -> float:
        """Estimates sigma_1^2 = ||H||_2^2 via power iteration on H^T H.

        Returns:
            sigma_1^2  (squared spectral norm).
        """
        assert grad_fv is not None

        hvp = self._hvp_closure(grad_fw, f_w_graph, f_v)  # H_avg z   (f_w -> f_v)
        hvp_adj = self._hvp_closure(grad_fv, f_v, f_w_graph)  # H_avg^T u (f_v -> f_w)
        return power_iteration_spectral_sq(
            hvp, hvp_adj, d_w, self._n_power_iter, device, generator=self._generator
        )

    def _hutchinson_shared(
        self,
        f_v: Tensor,
        f_w_graph: Tensor,
        grad_fw: Tensor,
        d_w: int,
        device: torch.device,
    ) -> float:
        """Hutchinson estimator with shared probe vectors.

        Returns:
            Estimate of ||H_avg||_F^2.
        """
        hvp = self._hvp_closure(grad_fw, f_w_graph, f_v)
        return hutchinson_frob_sq(
            hvp, d_w, self._n_probes, device, generator=self._generator
        )

    # ------------------------------------------------------------------
    # Combined estimator (avoids double graph setup / Hutchinson)
    # ------------------------------------------------------------------

    def estimate_frob_and_stable_rank(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> tuple[float, float]:
        """Returns (||H||_F, D) in a single graph pass: shared Hutchinson + power iter.

        Avoids rebuilding the graph and saves m additional HVP calls.
        """
        f_v, f_w_graph, grad_fw, grad_fv = self._setup_graph(
            x,
            y,
            layer_v,
            layer_w,
            need_grad_fv=True,
        )

        device = f_v.device
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        frob_sq = self._hutchinson_shared(f_v, f_w_graph, grad_fw, d_w, device)
        frob_norm = frob_sq**0.5

        spectral_sq = self._power_iteration(
            f_v,
            f_w_graph,
            grad_fw,
            grad_fv,
            d_w,
            device,
        )
        stable_rank = frob_sq / spectral_sq if spectral_sq > 1e-24 else math.nan

        return frob_norm, stable_rank

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def estimate_resonance(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """R(v,w) = ||H^f_{v,w}||_F."""
        return self.estimate_block_frob_norm(x, y, layer_v, layer_w)

    def estimate_coupling(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """C(v,w) = ||H^f_{v,w}||_F / sqrt(||H^f_{v,v}||_F * ||H^f_{w,w}||_F)."""
        r_vw = self.estimate_block_frob_norm(x, y, layer_v, layer_w)
        r_vv = self.estimate_block_frob_norm(x, y, layer_v, layer_v)
        r_ww = self.estimate_block_frob_norm(x, y, layer_w, layer_w)
        return coupling_from_norms(r_vw, r_vv, r_ww)
