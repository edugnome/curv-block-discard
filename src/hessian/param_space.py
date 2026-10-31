"""
Parameter-space inter-layer Hessian blocks H_{theta_v, theta_w} and the
discarded-coupling fraction Phi.

Where the activation-space estimator (`stochastic.StochasticHessianEstimator`)
measures curvature between *activations* f_v, f_w, this module measures curvature
between *parameter groups* theta_v, theta_w. Phi quantifies the exact Hessian's
off-group mass for that partition; K-FAC / EKFAC also approximate within-block
curvature and may use a different partition.

Parameter Hessian block (exact action, one double-backward):
  (H u)_{theta_v} = grad_{theta_v} < grad_{theta_w} L, u_{theta_w} >
                  = H_{theta_v, theta_w} u_{theta_w}.

Block Frobenius norm via Hutchinson (no full matrix):
  ||H_{theta_v, theta_w}||_F^2 = E_z[ || H_{theta_v, theta_w} z ||^2 ],  z ~ Rademacher(d_w).

Discarded-coupling fraction (the fraction of total curvature mass any
block-diagonal preconditioner discards):
  Phi = ( sum_{v != w} ||H_{theta_v, theta_w}||_F^2 ) / ||H||_F^2
      = 1 - ( sum_v ||H_{theta_v, theta_v}||_F^2 ) / ||H||_F^2   in [0, 1] for nonzero H.

Estimator identity: ||H||_F^2 from a full-parameter Rademacher probe; each diagonal
||H_{theta_v,theta_v}||_F^2 from a probe supported only on group v - all sharing one
backward graph g = grad_theta L (create_graph=True), so the second backward
differentiates that single graph. The block probes are batched (`is_grads_batched`);
the full-parameter probe is the one allocation that scales with the whole model, so
it is drawn one probe at a time and never as a P-vector (`_LeanFullProbe`).

The same single-graph machinery backs `curvature(x, y)`: the per-minibatch bundle
(detached group-major gradient, exact full-parameter HVP, and a deferred inter-block
coupling) that a parameter-space second-order step (COUPLE-FAC) consumes.

Complexity: O((L+1) m) HVPs for Phi (L groups + full), O(P) memory.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle

from ._estimators import HVPClosure, rademacher_batch, rademacher_signs
from .memory import measure_peak
from .metrics import coupling_from_norms
from .types import LayerID

_EPS = 1e-12
_SEED_HIGH = 2**62  # per-tensor probe seeds are int64 draws below this bound

# Batched *block*-probe memory control (the full-parameter probe is `_LeanFullProbe`).
# The vmapped double-backward of a probe chunk carries
# (chunk x backward-intermediate) buffers that scale with the minibatch and the
# activations, not with the parameter dimensions, so they cannot be sized analytically
# - and sizing the chunk to whatever memory happened to be free let it fill the card
# (a 12 GiB peak with 12 GiB free, 22.7 GiB with 23 free) and starve every allocation
# outside the probe loop: a Newton-CG oracle HVP, a cuSOLVER handle, a K-FAC Cholesky.
# The chunk is therefore *calibrated*: the first probe of every call runs alone and its
# measured transient peak (`memory.measure_peak`) is the per-probe cost, and the chunk
# is that cost divided into a fixed fraction of the device's *total* memory. Sizing
# against the total rather than the free memory keeps the chunk - and the run's peak -
# the same whatever else shares the card, and the fraction is the headroom left to the
# resident set and to everything that runs after the probes. The OutOfMemoryError
# back-off in `_frob_sq_batched` (halve the chunk, down to single probes) stays as the
# last resort. On CPU, which has no peak counter, a fixed element budget is used.
_PROBE_MEM_FRACTION = 0.25  # of the device total, for one chunk's transient peak
_PROBE_ELEM_BUDGET = 1 << 25  # CPU: ~33.5M elements (~128 MB fp32) per chunk


class ParamGroupedModel(Protocol):
    """A model whose trainable parameters are partitioned into named groups.

    Groups are the parameter-space analogue of the activation measurement points
    in `SegmentedModel`. The model must be callable (x -> logits) so the loss graph
    can be built directly in parameter space.

    Two members are all this protocol asks for, so a caller that also needs `nn.Module`
    behaviour - walking submodules, toggling train/eval, handing the model to a library
    whose signature names the concrete class, as curvlinops does - asserts the concrete
    type where it needs it (`cast(nn.Module, model)`) instead of widening the protocol.
    Structural typing cannot express "is an nn.Module", and requiring it here would make
    every model in the codebase inherit from a base class of ours to be measurable.
    """

    def __call__(self, x: Tensor) -> Tensor: ...
    def get_param_groups(self) -> dict[LayerID, list[Parameter]]: ...


@dataclass(frozen=True)
class ParamCurvature:
    """Per-minibatch curvature bundle for one parameter-space second-order step.

    The members share the single backward graph g = grad_theta L (create_graph=True)
    the estimator builds, so they are mutually consistent and cheap to obtain together;
    holding this object keeps that graph alive (discard it after the step to free it).

      - ``loss``: the minibatch loss L(theta) the graph was built on - read off that
        forward, so a runner tracing the loss along a trajectory spends no second
        forward (which, in train mode, would also update the BatchNorm statistics);
      - ``flat_grad``: the *detached* parameter gradient, concatenated in
        `get_param_groups` (group-major) order - the layout a K-FAC provider's
        `block_ranges` and the optimizer's parameter list must match;
      - ``hvp``: the exact full-parameter Hessian-vector product z |-> H z;
      - ``coupling``: a *deferred* call returning the inter-block coupling C(v, w) for
        every pair v before w in group order. It costs one set of Hutchinson probes per
        block, so it is kept behind a call for the runner to amortize over T steps.
    """

    loss: float
    flat_grad: Tensor
    hvp: HVPClosure
    coupling: Callable[[], dict[tuple[LayerID, LayerID], float]]


@dataclass(frozen=True)
class PhiReport:
    """Parameter-space diagnostic for one minibatch: Phi and block norms.

    The three members come from the single backward graph g = grad_theta L the estimator
    builds, sharing probe work where applicable. The individual block estimates remain
    stochastic (the optimization counterpart is `ParamCurvature`):

      - ``phi``: estimated off-block fraction (NaN if the denominator is near zero).
        Independent full and diagonal estimates can yield a negative value;
      - ``diag_frob``: per-block ||H_{theta_v, theta_v}||_F, locating the diagonal mass
        of the exact Hessian, in group-major order;
      - ``coupling``: C(v, w) for every pair v before w in group order, locating the
        off-diagonal mass for the requested pairs.
    """

    phi: float
    diag_frob: dict[LayerID, float]
    coupling: dict[tuple[LayerID, LayerID], float]


@dataclass
class _ParamGraph:
    """A single backward graph g = grad_theta L, grouped by parameter block.

    Holds everything the HVP closures need so the graph is built once and reused
    across all probes / blocks (the second backward differentiates this graph).
    """

    names: list[LayerID]
    params: dict[LayerID, list[Parameter]]
    grads: dict[LayerID, list[Tensor]]
    dims: dict[LayerID, int]
    device: torch.device
    loss: float  # L(theta) on the minibatch the graph was built on

    @property
    def all_params(self) -> list[Parameter]:
        return [p for name in self.names for p in self.params[name]]

    @property
    def all_grads(self) -> list[Tensor]:
        return [g for name in self.names for g in self.grads[name]]

    @property
    def total_dim(self) -> int:
        return sum(self.dims[name] for name in self.names)


class _LeanFullProbe:
    """One full-parameter Hutchinson probe ||H z||^2 that never holds a P-vector.

    The batched block probe (`ParamBlockEstimator._frob_sq_batched`) materialises, per
    probe, the concatenated gradient, the probe (int8 bits + float) and the returned
    grads: ~3.25 P bytes on top of the resident set, 4.7 P measured on Qwen2.5-0.5B
    once the vmapped transient is counted - which is what put a 1.2B-parameter model
    out of reach of a 24 GB card. Here the probe is *injected lazily* and the result
    *drained eagerly*, so nothing of size P exists at once:

      - the roots of the second backward are the per-tensor gradients g_i themselves,
        seeded with stride-0 zero views (no allocation). A tensor pre-hook on every g_i
        swaps that seed for the tensor's Rademacher slice z_i - drawn from the tensor's
        own seed, int8 bits -> the gradient's dtype - the moment ``g_i.grad_fn`` runs,
        and the engine drops the slice once that node has consumed it;
      - ``torch.autograd.backward(inputs=params)`` accumulates each slice (H z)_i into
        ``p.grad``, where a post-accumulate-grad hook adds ||(H z)_i||^2 to the running
        total and clears ``p.grad`` in the same breath, so every slice dies as it lands.

    What remains at the peak is the engine's own in-flight state - partial sums waiting
    for their last contribution (bounded by P), the activation-sized double-backward
    buffers, and one tensor's probe slice: 1.2 P measured against 3.2 P for the batched
    probe on an MLP (`tests/test_lean_probe.py`). A causal LM with a *tied* embedding
    pays for two more copies of that one matrix (its slice waits for its second consumer,
    the lm_head branch, while two V x d contributions accumulate) plus ~0.55 GB of
    activation transient per sequence of 128 tokens: extrapolated to Llama-3.2-1B in
    fp32 that is 8.7 GiB over the resident set at 4 x 128 tokens (7.6 GiB at 2 x 128)
    against ~23 GiB for the batched probe.

    Hutchinson estimator and expectation are unchanged; only the RNG stream differs.
    The per-tensor seeds come from the caller (drawn in a fixed, group-major order from
    one generator), so a seed fixes the estimate whatever order the engine visits the
    tensors in. A gradient with no graph of its own (constant in theta) has a vanishing
    column of H and is left out of the roots. Existing ``.grad`` values are stashed and
    restored and every hook is removed on exit, so the model is left as it was found.
    """

    def __init__(
        self, params: list[Parameter], grads: list[Tensor], device: torch.device
    ) -> None:
        self._params = params
        self._grads = [g for g in grads if g.requires_grad]
        self._device = device
        self._seeds: list[int] = []
        self._total = torch.zeros((), device=device)
        self._handles: list[RemovableHandle] = []
        self._stashed: list[Tensor | None] = []

    @property
    def n_slices(self) -> int:
        """Number of probe slices (= seeds) one probe needs."""
        return len(self._grads)

    def __enter__(self) -> _LeanFullProbe:
        self._stashed = [p.grad for p in self._params]
        for p in self._params:
            p.grad = None
            self._handles.append(p.register_post_accumulate_grad_hook(self._drain))
        for i, g in enumerate(self._grads):
            self._handles.append(g.register_hook(self._injector(i, g)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for p, g in zip(self._params, self._stashed, strict=True):
            p.grad = g

    def __call__(self, seeds: list[int]) -> Tensor:
        """||H z||^2 for the probe whose slice i is Rademacher(seeds[i]); 0-d tensor."""
        if len(seeds) != self.n_slices:
            raise ValueError(f"expected {self.n_slices} seeds, got {len(seeds)}")
        self._seeds = seeds
        self._total = torch.zeros((), device=self._device)
        if self._grads:
            torch.autograd.backward(
                self._grads,
                grad_tensors=[g.new_zeros(()).expand(g.shape) for g in self._grads],
                inputs=self._params,
                retain_graph=True,
            )
        return self._total

    def _drain(self, p: Tensor) -> None:
        """Post-accumulate hook: fold ||(H z)_p||^2 into the total, free the slice."""
        g = p.grad
        if g is not None:
            with torch.no_grad():
                self._total = self._total + torch.linalg.vector_norm(g).square().float()
            p.grad = None

    def _injector(self, index: int, g: Tensor) -> Callable[[Tensor], Tensor]:
        """Pre-hook on g_i's node: replace the zero seed with the slice z_i, just in time."""
        shape, dtype = tuple(g.shape), g.dtype

        def inject(_seed: Tensor) -> Tensor:
            gen = torch.Generator(device=self._device).manual_seed(self._seeds[index])
            return rademacher_signs(shape, self._device, gen, dtype)

        return inject


class ParamBlockEstimator:
    """Hutchinson estimation of parameter-space blocks H_{theta_v, theta_w} and Phi.

    ``generator`` (optional) drives every Hutchinson draw - the batched block probes and
    the per-tensor seeds of the full-parameter probe - so a seeded generator fixes the
    estimates; it must live on the parameters' device. ``None`` uses that device's
    global RNG (``torch.manual_seed`` then fixes the estimates).
    """

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_probes: int = 30,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._n_probes = n_probes
        self._generator = generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_block_frob_norm(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """||H_{theta_v, theta_w}||_F via Hutchinson on the cross-block HVP."""
        graph = self._grad_graph(x, y)
        return self._block_frob_sq(graph, layer_v, layer_w) ** 0.5

    def estimate_full_frob_sq(self, x: Tensor, y: Tensor) -> float:
        """||H||_F^2 (full parameter Hessian) via Hutchinson."""
        return self._full_frob_sq(self._grad_graph(x, y))

    def estimate_diag_frob_sq(self, x: Tensor, y: Tensor, layer_v: LayerID) -> float:
        """||H_{theta_v, theta_v}||_F^2 (a single diagonal block)."""
        return self._block_frob_sq(self._grad_graph(x, y), layer_v, layer_v)

    def estimate_phi(self, x: Tensor, y: Tensor) -> float:
        """Estimate Phi = 1 - sum_v ||H_{vv}||_F^2 / ||H||_F^2.

        Independent estimates can give negative Phi; a near-zero denominator gives NaN.

        Built on a single backward graph: one full-parameter probe set for ||H||_F^2
        and one group-localized probe set per diagonal block.
        """
        graph = self._grad_graph(x, y)
        full_sq = self._full_frob_sq(graph)
        if full_sq < _EPS:
            return math.nan
        diag_sq = sum(self._block_frob_sq(graph, name, name) for name in graph.names)
        return 1.0 - diag_sq / full_sq

    def estimate_phi_report(
        self,
        x: Tensor,
        y: Tensor,
        *,
        coupling_pairs: Iterable[tuple[LayerID, LayerID]] | None = None,
    ) -> PhiReport:
        """Phi plus its diagonal-mass and coupling localization, from one backward graph.

        One group-localized probe set per block gives every diagonal norm; the
        off-diagonal norms give the coupling field C(v, w). Each diagonal
        block is measured once and reused for both Phi and the couplings that contain it
        - the same single-graph amortization as `curvature`'s deferred ``coupling``,
        exposed here for measurement rather than for a step.

        ``coupling_pairs`` restricts the (always group-major) coupling field to a caller-
        chosen subset of (v, w) pairs; ``None`` measures every v-before-w pair. When all
        off-diagonal blocks are measured, ``||H||_F^2`` is estimated using the block
        decomposition ``sum_v ||H_vv||_F^2 + 2 sum_{v<w} ||H_vw||_F^2`` (using
        ``||H_vw||_F = ||H_wv||_F``), so no separate full-parameter probe is drawn - it is
        the single largest allocation. For finite nonnegative block estimates and a
        nonzero denominator, Phi is in [0, 1]. A subset
        (e.g. a depth band for a many-block transformer) leaves that sum incomplete, so
        ``||H||_F^2`` then falls back to a full-parameter Hutchinson probe.
        """
        graph = self._grad_graph(x, y)
        diag_sq = {v: self._block_frob_sq(graph, v, v) for v in graph.names}
        diag_frob = {v: s**0.5 for v, s in diag_sq.items()}

        if coupling_pairs is None:
            names = graph.names
            pairs: Iterable[tuple[LayerID, LayerID]] = [
                (v, names[j])
                for i, v in enumerate(names)
                for j in range(i + 1, len(names))
            ]
        else:
            pairs = coupling_pairs
        coupling: dict[tuple[LayerID, LayerID], float] = {}
        offdiag_sq = 0.0
        for v, w in pairs:
            block_sq = self._block_frob_sq(graph, v, w)
            offdiag_sq += block_sq
            coupling[(v, w)] = coupling_from_norms(
                block_sq**0.5, diag_frob[v], diag_frob[w]
            )

        diag_total = sum(diag_sq.values())
        if coupling_pairs is None:
            # Every off-diagonal block measured: the exact block decomposition of
            # ||H||_F^2 using estimated block norms; no separate full probe is needed.
            full_sq = diag_total + 2.0 * offdiag_sq
        else:
            full_sq = self._full_frob_sq(graph)
        phi = math.nan if full_sq < _EPS else 1.0 - diag_total / full_sq

        return PhiReport(phi=phi, diag_frob=diag_frob, coupling=coupling)

    def estimate_coupling(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """C(v,w) = ||H_{vw}||_F / sqrt(||H_{vv}||_F ||H_{ww}||_F) in parameter space.

        The three blocks share one backward graph. As in activation space, C > 1 on at
        least one pair is possible only if H is not positive semi-definite (see
        `hessian.metrics.coupling_from_norms`); it is left unclamped.
        """
        graph = self._grad_graph(x, y)
        r_vw = self._block_frob_sq(graph, layer_v, layer_w) ** 0.5
        r_vv = self._block_frob_sq(graph, layer_v, layer_v) ** 0.5
        r_ww = self._block_frob_sq(graph, layer_w, layer_w) ** 0.5
        return coupling_from_norms(r_vw, r_vv, r_ww)

    def curvature(self, x: Tensor, y: Tensor) -> ParamCurvature:
        """Per-minibatch curvature bundle feeding one COUPLE-FAC step.

        Builds the single backward graph g = grad_theta L (create_graph=True) once and
        exposes, all sharing it: the detached group-major gradient ``flat_grad``; the
        exact full-parameter HVP ``hvp`` (z |-> H z); and ``coupling`` - a deferred call
        returning the inter-block coupling C(v, w), kept lazy because it costs one set of
        Hutchinson probes per block and the caller amortizes it over T steps.
        """
        graph = self._grad_graph(x, y)
        flat_grad = torch.cat([g.reshape(-1) for g in graph.all_grads]).detach()
        hvp = self._cross_block_hvp(graph.all_params, graph.all_grads)
        return ParamCurvature(
            loss=graph.loss,
            flat_grad=flat_grad,
            hvp=hvp,
            coupling=lambda: self._coupling_dict(graph),
        )

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _grad_graph(self, x: Tensor, y: Tensor) -> _ParamGraph:
        """Builds g = grad_theta L (create_graph=True), grouped by parameter block."""
        groups = self._model.get_param_groups()
        names = list(groups.keys())
        params = {name: list(groups[name]) for name in names}

        flat_params = [p for name in names for p in params[name]]
        loss = self._loss_fn(self._model(x), y)
        flat_grads = torch.autograd.grad(loss, flat_params, create_graph=True)

        grads: dict[LayerID, list[Tensor]] = {}
        dims: dict[LayerID, int] = {}
        offset = 0
        for name in names:
            k = len(params[name])
            grads[name] = list(flat_grads[offset : offset + k])
            dims[name] = sum(p.numel() for p in params[name])
            offset += k

        return _ParamGraph(
            names=names,
            params=params,
            grads=grads,
            dims=dims,
            device=flat_params[0].device,
            loss=loss.item(),
        )

    # ------------------------------------------------------------------
    # Block estimators
    # ------------------------------------------------------------------

    def _block_frob_sq(self, graph: _ParamGraph, v: LayerID, w: LayerID) -> float:
        """||H_{theta_v, theta_w}||_F^2 via batched Hutchinson on the cross-block HVP."""
        return self._frob_sq_batched(graph.params[v], graph.grads[w], graph.dims[w])

    def _full_frob_sq(self, graph: _ParamGraph) -> float:
        """||H||_F^2 via Hutchinson on the full-parameter HVP, one lean probe at a time.

        The full probe is the single allocation that scales with the whole model, so
        it is not batched: each probe is a `_LeanFullProbe` pass (no P-vector beyond
        the engine's in-flight state) with its per-tensor seeds drawn from the
        estimator's generator in group-major order.
        """
        total = torch.zeros((), device=graph.device)
        with _LeanFullProbe(graph.all_params, graph.all_grads, graph.device) as probe:
            for _ in range(self._n_probes):
                total = total + probe(self._probe_seeds(probe.n_slices, graph.device))
        return (total / self._n_probes).item()

    def _probe_seeds(self, n: int, device: torch.device) -> list[int]:
        """``n`` per-tensor seeds for one lean probe, from the generator or device RNG."""
        gen = self._generator
        where = device if gen is None else gen.device
        return torch.randint(0, _SEED_HIGH, (n,), device=where, generator=gen).tolist()

    def _frob_sq_batched(
        self,
        out_params: list[Parameter],
        grad_probe: list[Tensor],
        dim_probe: int,
    ) -> float:
        """||H_{out, probe}||_F^2 via Hutchinson with *batched* Rademacher probes.

        The block path (diagonal and cross blocks, whose probe spans one group). One
        vectorized double-backward (``is_grads_batched``) replaces the per-probe Python
        loop: the probes are stacked into (m, dim_probe) and pushed through the retained
        graph at once. The first probe runs alone and calibrates ``m`` from its measured
        transient peak (see `_probe_chunk`); an OutOfMemoryError halves the chunk, down
        to single probes, before giving up. ``allow_unused`` coordinates return None and
        contribute zero to the sum of squares, so they are skipped.
        """
        flat_grad = torch.cat([g.reshape(-1) for g in grad_probe])
        device = flat_grad.device
        per_probe_elems = dim_probe + sum(p.numel() for p in out_params)

        total = torch.zeros((), device=device)
        drawn = 0
        chunk = 1  # the calibration probe
        while drawn < self._n_probes:
            m = min(chunk, self._n_probes - drawn)
            try:
                with measure_peak(device) as peak:
                    probes = rademacher_batch(m, dim_probe, device, self._generator)
                    grads = torch.autograd.grad(
                        flat_grad,
                        out_params,
                        grad_outputs=probes,
                        retain_graph=True,
                        allow_unused=True,
                        is_grads_batched=True,
                    )
            except torch.cuda.OutOfMemoryError:
                if m == 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, m // 2)
                continue
            if drawn == 0:
                chunk = self._probe_chunk(peak[0], per_probe_elems, device)
            for g in grads:
                if g is not None:
                    total = total + (g.reshape(m, -1) ** 2).sum()
            drawn += m
            del probes, grads  # not resident while the next chunk runs
        return (total / self._n_probes).item()

    def _probe_chunk(
        self, probe_bytes: int, per_probe_elems: int, device: torch.device
    ) -> int:
        """Probes per batched double-backward, from one probe's measured footprint.

        On CUDA ``probe_bytes`` is the transient peak the single calibration probe
        needed (its probe vector, its grad slice and the vmapped backward buffers);
        the chunk is `_PROBE_MEM_FRACTION` of the device's total memory divided by
        it. On CPU there is no peak counter, so ``per_probe_elems`` - the probe vector
        plus its grad slice - is budgeted against `_PROBE_ELEM_BUDGET`. The result is
        clamped to [1, n_probes].
        """
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            chunk = int(total * _PROBE_MEM_FRACTION) // max(1, probe_bytes)
        else:
            chunk = _PROBE_ELEM_BUDGET // max(1, per_probe_elems)
        return max(1, min(self._n_probes, chunk))

    def _coupling_dict(
        self, graph: _ParamGraph
    ) -> dict[tuple[LayerID, LayerID], float]:
        """All inter-block couplings C(v, w), v before w in group order, from one graph.

        Each diagonal norm is measured once and reused across the pairs that contain it -
        cheaper than `estimate_coupling`, which re-measures both diagonals per pair (and
        rebuilds the graph).
        """
        names = graph.names
        diag = {v: self._block_frob_sq(graph, v, v) ** 0.5 for v in names}
        coupling: dict[tuple[LayerID, LayerID], float] = {}
        for i, v in enumerate(names):
            for w in names[i + 1 :]:
                r_vw = self._block_frob_sq(graph, v, w) ** 0.5
                coupling[(v, w)] = coupling_from_norms(r_vw, diag[v], diag[w])
        return coupling

    @staticmethod
    def _cross_block_hvp(
        out_params: list[Parameter],
        grad_probe: list[Tensor],
    ) -> HVPClosure:
        """Builds z |-> (H z) restricted to ``out_params`` .

        z is a flat Rademacher probe over group w, the block behind ``grad_probe``
        (= dL/d theta_w); the returned vector is the ``out_params`` (group v) slice
        of H z:
          grad_{out_params} < grad_probe, z > = H_{out, w} z.

        Specializations (one helper, three uses):
          - cross block:  out = theta_v, grad_probe = dL/d theta_w;
          - diagonal:     out = theta_v, grad_probe = dL/d theta_v;
          - full HVP:     out = all params, grad_probe = dL/d theta.

        ``allow_unused=True`` zero-fills structurally disconnected coordinates, which
        is exactly their (vanishing) second-order contribution.
        """
        flat_grad = torch.cat([g.reshape(-1) for g in grad_probe])

        def hvp(z: Tensor) -> Tensor:
            dot = flat_grad.dot(z)
            grads = torch.autograd.grad(
                dot, out_params, retain_graph=True, allow_unused=True
            )
            return torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, out_params, strict=True)
                ]
            )

        return hvp
