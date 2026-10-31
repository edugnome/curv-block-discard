"""
CurvlinopsKFACProvider: the concrete K-FAC backend behind the `KFACProvider` seam.

This is the single place in `secondorder/` that binds a real K-FAC library
(curvlinops, Dangel et al.). It supplies the block-diagonal Newton / natural-gradient
direction the overlay then corrects:

  d0 = -( blkdiag_l (F_l + lambda I) )^{-1} g    over K-FAC-able layers,
  d0 = -g                                        for every other parameter,

where each per-layer block F_l is the Kronecker-factored curvature F_l = G_l (x) A_l
(output factor G_l = E[delta delta^T], input factor A_l = E[a a^T]) that curvlinops
estimates by hooking the forward / backward pass of every supported layer. Because the
factors are collected by *local* per-layer hooks - not by tracing the module graph -
the provider is architecture-agnostic: skip connections, BatchNorm inside the forward,
and custom ``forward`` methods are transparent, so the *whole* network is evaluated
(graph-tracing K-FAC, by contrast, is limited to plain sequential stacks).

Supported layers are ``Linear`` and ``Conv2d``. A parameter outside them (e.g. a
BatchNorm / LayerNorm affine weight) carries no Kronecker curvature and receives the
identity step ``-g`` - the well-defined limit of a damped Newton step with no curvature.
The split is per *parameter*, so a group mixing a convolution with its normalisation is
handled correctly.

The per-layer block solve uses curvlinops' *factored* Tikhonov damping: each Kronecker
factor is damped and Cholesky-inverted on its own,

  d0|_l = -(G_l + lambda I)^{-1} grad_W_l (A_l + lambda I)^{-1},

the canonical K-FAC update (Martens & Grosse, 2015, Section 6.3). This is the damped
block-diagonal natural-gradient step the overlay corrects. Factoring the damping avoids
the eigendecomposition of the *raw* factors G_l, A_l that exact (joint-eigenbasis)
damping needs - an ``eigh`` that fails to converge in float32 on the rank-deficient
input factors of a deep convolutional network (A_l is C_in k^2 wide but has rank at most
batch * spatial) - and curvlinops retries a factor's Cholesky in float64 should it be
numerically singular. For a single linear layer K-FAC *is* the exact GGN (G_l (x) A_l),
so as lambda -> 0 the step converges to the exact GGN Newton step -(GGN)^{-1} g (the
adapter test asserts this lambda -> 0 limit); the factored and joint dampings differ only
in how the finite-lambda Tikhonov term is distributed. With a bias or a multi-class
cross-entropy, K-FAC is - as always - an approximation, which is precisely what the
coupling-gated overlay and the no-harm safeguard exist to handle.

The flat parameter layout - and therefore ``block_ranges`` - is the group-major,
parameter-major concatenation of ``model.get_param_groups()``, identical to the layout
the parameter-space estimator (`hessian.param_space`) and the overlay's HVP operate on,
so the base direction, the coupling, and the correction all compose. The K-FAC-able
parameters are handed to curvlinops in that same order, so the operator's vector layout
matches the gradient's supported sub-vector.

Both providers amortize their curvature over a refresh period T_k (`_PeriodicRefresh`):
``update`` is called once per step, the Kronecker factors and their inverse (K-FAC) or
the eigenbasis (EKFAC) are rebuilt only on the calls with ``calls % T_k == 0`` - the
first call included - and the gradient is fresh every step,

  d0_t = -( K^_{floor(t / T_k) * T_k} + lambda I )^{-1} g_t,

the standard periodic factor refresh of K-FAC (Martens & Grosse, 2015). The damping is
read at each refresh (``float(damping)``), so a run whose lambda the runner adapts
(`secondorder.damping.Damping`) rebuilds its inverse with the lambda then in force -
the same one the overlay, the safeguard and the oracle read - rather than the initial one.
"""

from __future__ import annotations

import logging
import time
from typing import cast

import torch
import torch.nn as nn
from curvlinops import (
    EKFACLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
)
from torch import Tensor
from torch.nn import Parameter

from hessian.param_space import ParamGroupedModel
from hessian.types import LayerID

from .damping import Damping

logger = logging.getLogger(__name__)

# curvlinops K-FAC factorises these layer types; any other parameter (norm affine
# weights, etc.) has no Kronecker curvature and falls back to the identity step.
_SUPPORTED: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d)


def _build_ranges(groups: dict[LayerID, list[Parameter]]) -> dict[LayerID, slice]:
    """Group-major slices of the flat parameter vector (matches the estimator layout)."""
    ranges: dict[LayerID, slice] = {}
    offset = 0
    for name, params in groups.items():
        size = sum(p.numel() for p in params)
        ranges[name] = slice(offset, offset + size)
        offset += size
    return ranges


def _classify_supported(
    model: ParamGroupedModel, groups: dict[LayerID, list[Parameter]]
) -> tuple[list[Parameter], Tensor]:
    """Split the group-major flat layout into curvature-able params and a support mask.

    The ``Linear`` / ``Conv2d`` parameters - the ones carrying Kronecker curvature - are
    returned in group-major order (the layout curvlinops receives), and ``sup_mask`` flags
    their positions in the flat parameter vector so ``base_direction`` routes the curvature
    step to them and the identity step ``-g`` to every other parameter (norm affines, etc.).
    ``model`` is walked as an ``nn.Module`` to classify the modules owning each parameter.
    """
    supported_ids = {
        id(p)
        for module in cast(nn.Module, model).modules()
        if isinstance(module, _SUPPORTED)
        for p in module.parameters(recurse=False)
    }
    total = sum(p.numel() for params in groups.values() for p in params)
    sup_mask = torch.zeros(total, dtype=torch.bool)
    kfac_params: list[Parameter] = []
    offset = 0
    for params in groups.values():
        for p in params:
            n = p.numel()
            if id(p) in supported_ids:
                kfac_params.append(p)
                sup_mask[offset : offset + n] = True
            offset += n
    return kfac_params, sup_mask


class _PeriodicRefresh:
    """Refresh-period bookkeeping shared by the curvature providers.

    ``update`` is called once per optimizer step; the curvature is rebuilt only on the
    calls with ``calls % refresh_period == 0`` (so on the very first call) and reused in
    between. ``n_refreshes`` counts the rebuilds that completed.
    """

    def __init__(self, refresh_period: int) -> None:
        self._refresh_period = max(1, refresh_period)
        self._calls = 0
        self._n_refreshes = 0

    @property
    def n_refreshes(self) -> int:
        """Curvature rebuilds completed so far."""
        return self._n_refreshes

    def _tick(self) -> bool:
        """Count one ``update`` call; True when it is a scheduled refresh."""
        due = self._calls % self._refresh_period == 0
        self._calls += 1
        return due


class CurvlinopsKFACProvider(_PeriodicRefresh):
    """K-FAC provider backed by curvlinops (structurally implements `KFACProvider`).

    Stateful by design: ``update(x, y)`` is called every step and, every
    ``refresh_period`` updates, rebuilds the Kronecker-factored operator on that batch and
    collects its factors right away (a forward/backward pass whose activations are freed
    before the step's HVP graph exists); their damped Cholesky inverse is then computed
    and cached by curvlinops on the first ``base_direction`` that needs it.
    ``base_direction(g)`` returns the damped block-diagonal Newton step of the last
    refresh's curvature for the fresh gradient.
    The model is used as-is - the caller owns its mode (e.g. BatchNorm in eval during
    factor collection) and its gradients (``base_direction`` never touches the autograd
    graph).
    """

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: nn.Module,
        *,
        damping: float | Damping = 1e-2,
        fisher_type: str = "type-2",
        refresh_period: int = 1,
    ) -> None:
        super().__init__(refresh_period)
        self._model = model
        self._loss_fn = loss_fn
        self._damping = damping  # read at every refresh, never copied
        self._fisher_type = fisher_type

        self._groups = model.get_param_groups()
        self._ranges = _build_ranges(self._groups)

        # Per-parameter split (in flat-layout order): the Linear / Conv2d parameters go
        # to curvlinops; the rest take the identity step.
        self._kfac_params, self._sup_mask = _classify_supported(model, self._groups)

        self._inverse: KFACInverseLinearOperator | None = None
        self._ready = False

    # -- KFACProvider --------------------------------------------------
    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(blkdiag_l (F_l + lambda I))^{-1} grad on K-FAC-able layers, -grad else."""
        if not self._ready:
            raise RuntimeError("call update(x, y) before base_direction(grad)")
        grad = grad.detach()
        d0 = -grad  # identity (gradient-descent) step ...
        if self._inverse is not None:
            if self._sup_mask.device != grad.device:
                self._sup_mask = self._sup_mask.to(grad.device)
            mask = self._sup_mask
            try:
                d0[mask] = -(self._inverse @ grad[mask])  # ... K-FAC where it applies
            except torch.linalg.LinAlgError:
                # A degenerate Kronecker factor (non-positive-definite Cholesky) from the
                # last refresh: take the identity step -grad on the K-FAC-able params
                # rather than aborting the run (the no-harm safeguard, when active,
                # backstops it) until the next refresh rebuilds the factors.
                logger.warning(
                    "K-FAC factor inversion failed (non-positive-definite factor); "
                    "taking the gradient step on this minibatch."
                )
        return d0

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each group."""
        return self._ranges

    # -- curvature refresh ---------------------------------------------
    def update(self, x: Tensor, y: Tensor) -> CurvlinopsKFACProvider:
        """Rebuild the Kronecker-factored inverse on (x, y) every ``refresh_period``."""
        if self._tick() and self._kfac_params:
            operator = KFACLinearOperator(
                self._model,
                self._loss_fn,
                self._kfac_params,
                [(x, y)],
                fisher_type=self._fisher_type,
                check_deterministic=False,
            )
            # Collect factors before building the step's HVP graph so that the
            # factor pass's activations do not overlap with the HVP graph in memory.
            operator.compute_kronecker_factors()
            # Factored Tikhonov damping (Cholesky on each damped factor, with a
            # float64 retry on a singular factor); the exact joint-eigenbasis damping
            # would instead run ``eigh`` on the raw factors, which fails to converge in
            # float32 on a deep conv net's rank-deficient input factors.
            self._inverse = KFACInverseLinearOperator(
                operator, damping=float(self._damping)
            )
            self._n_refreshes += 1
        self._ready = True
        return self


class _DoubleEigenbasisEKFAC(EKFACLinearOperator):
    """curvlinops EKFAC whose Kronecker-factor eigendecomposition runs in float64.

    The factors and the eigenvalue correction are collected in the model's own dtype -
    the fast path: on the b1 ResNet-50 each is one forward plus, for the type-2 Fisher,
    one backward per class - and only the ``eigh`` of the factors is lifted to double
    precision, then the eigenvectors are cast back. That is the one step that fails to
    converge in float32 on the rank-deficient input factors of a convolutional network
    (``syevd`` on an ill-conditioned matrix), and it costs a few seconds either way.
    """

    def _compute_eigenvectors(self) -> None:
        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()
        for mod_name in self._mapping:
            for source, destination in (
                (self._input_covariances, self._input_covariances_eigenvectors),
                (self._gradient_covariances, self._gradient_covariances_eigenvectors),
            ):
                factor = source.pop(mod_name, None)
                if factor is not None:
                    eigenvectors = torch.linalg.eigh(factor.double()).eigenvectors
                    destination[mod_name] = eigenvectors.to(factor.dtype)


class CurvlinopsEKFACProvider(_PeriodicRefresh):
    """EKFAC provider backed by curvlinops (structurally implements `KFACProvider`).

    EKFAC (eigenvalue-corrected K-FAC; George et al., 2018) keeps the K-FAC Kronecker
    eigenbasis but rescales it by the exact diagonal variances measured in that basis - a
    strictly finer block-diagonal curvature than K-FAC, and the second block-diagonal
    baseline that demonstrates the overlay is optimizer-agnostic. It is wired exactly like
    `CurvlinopsKFACProvider` - the same group-major layout, the same Linear/Conv2d support
    split, the same identity step ``-g`` for norm parameters - with the differences the
    eigenvalue correction forces:

      * EKFAC supports only *exact* (eigenbasis) damping, so the inverse is built with
        ``use_exact_damping=True`` instead of the factored Tikhonov damping K-FAC uses;
      * that eigendecomposition of the raw Kronecker factors fails to converge in float32
        on the rank-deficient input factors of a convolutional network (``reduce``
        approximation: C_in k^2 wide, rank at most the batch size) - the same
        ill-conditioning that pushed K-FAC to factored damping. EKFAC cannot sidestep the
        eigenbasis, so `_DoubleEigenbasisEKFAC` runs that one ``eigh`` in float64. The
        factors and the correction themselves are collected in the model's dtype:
        collecting them instead on a private float64 copy of the model would, on a
        Turing card (1/32 rate double precision), turn the type-2 passes - two per
        refresh, one backward per class each - into hours, making the run look hung;
      * the factors are collected with the model in eval mode (BatchNorm on its running
        statistics, which the collection pass must not update): the deployed network's
        curvature. The caller's mode is restored afterwards.

    The eigenbasis is refreshed only every ``refresh_period`` updates and a refresh whose
    eigendecomposition does not converge keeps the previous eigenbasis rather than
    aborting the run (``n_refresh_failed`` counts them); every refresh logs its wall time.
    """

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: nn.Module,
        *,
        damping: float | Damping = 1e-2,
        fisher_type: str = "type-2",
        refresh_period: int = 1,
    ) -> None:
        super().__init__(refresh_period)
        self._model = model
        self._loss_fn = loss_fn
        self._damping = damping  # read at every refresh, never copied
        self._fisher_type = fisher_type
        self._n_refresh_failed = 0

        self._groups = model.get_param_groups()
        self._ranges = _build_ranges(self._groups)
        self._kfac_params, self._sup_mask = _classify_supported(model, self._groups)
        self._dim_supported = sum(p.numel() for p in self._kfac_params)

        self._inverse: KFACInverseLinearOperator | None = None
        self._ready = False

    # -- KFACProvider --------------------------------------------------
    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(EKFAC + lambda I)^{-1} grad on supported layers, -grad everywhere else."""
        if not self._ready:
            raise RuntimeError("call update(x, y) before base_direction(grad)")
        grad = grad.detach()
        d0 = -grad  # identity (gradient-descent) step ...
        if self._inverse is not None:
            if self._sup_mask.device != grad.device:
                self._sup_mask = self._sup_mask.to(grad.device)
            mask = self._sup_mask
            d0[mask] = -(self._inverse @ grad[mask])  # ... EKFAC where it applies
        return d0

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each group."""
        return self._ranges

    @property
    def n_refresh_failed(self) -> int:
        """Refreshes whose eigendecomposition did not converge (previous basis kept)."""
        return self._n_refresh_failed

    # -- curvature refresh ---------------------------------------------
    def update(self, x: Tensor, y: Tensor) -> CurvlinopsEKFACProvider:
        """Refresh the eigenvalue-corrected inverse every ``refresh_period`` updates."""
        if self._tick() and self._kfac_params:
            self._refresh(x, y)
        self._ready = True
        return self

    def _refresh(self, x: Tensor, y: Tensor) -> None:
        """Rebuild the eigenbasis on the current weights; keep the last one on failure."""
        started = time.perf_counter()
        # The eigenbasis builder below names nn.Module in its signature, so the
        # concrete type is asserted here; see `ParamGroupedModel`.
        module = cast(nn.Module, self._model)
        was_training = module.training
        module.eval()
        try:
            operator = _DoubleEigenbasisEKFAC(
                module,
                self._loss_fn,
                self._kfac_params,
                [(x, y)],
                fisher_type=self._fisher_type,
                kfac_approx="reduce",
                check_deterministic=False,
            )
            inverse = KFACInverseLinearOperator(
                operator, damping=float(self._damping), use_exact_damping=True
            )
            # Populate the factor, eigendecomposition and correction caches here;
            # a non-convergent factor then surfaces before base_direction.
            probe = torch.zeros(
                self._dim_supported,
                dtype=self._kfac_params[0].dtype,
                device=self._kfac_params[0].device,
            )
            _ = inverse @ probe
        except torch.linalg.LinAlgError:
            self._n_refresh_failed += 1
            logger.warning(
                "EKFAC eigendecomposition did not converge on this minibatch; keeping "
                "the previous eigenbasis (gradient step until the next successful build)."
            )
            return
        finally:
            module.train(was_training)
        self._inverse = inverse
        self._n_refreshes += 1
        logger.info(
            "EKFAC eigenbasis refresh %d took %.1f s",
            self._n_refreshes,
            time.perf_counter() - started,
        )
