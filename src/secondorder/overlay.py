"""
COUPLE-FAC overlay: coupling-gated rank-r correction over K-FAC.

Given a stored base direction d0 from a factored K-FAC/EKFAC preconditioner, the overlay:

  1. Selects coupled block pairs  S = {(v, w): C(v, w) > tau}.
  2. Builds ONE subspace over the selected pairs,
        U = orth( union over (v, w) in S of { d0|_v, d0|_w, top-r singular directions
                                              of H_{theta_v,theta_w} } ),
     and takes the *saddle-free reduced Newton* step there on the damped system,
        r  = g + (H + lambda I) d0          (current damped-system residual),
        c* = -(H^sf)^{-1} U^T r,     d1 = d0 + U c*.
     One joint solve, not one per pair: pairs sharing a block share the directions
     d0|_v, and independent per-pair solves would each take that step (the sum of two
     descent steps along one direction is not a descent step); on the joint basis the
     descent bound below applies to the whole correction.

The residual is not, in general, exactly the curvature discarded by K-FAC:
its factors are approximate and factored damping need not equal (bd(H)+lambda I).
Only when d0 solves the exact block-Hessian damped system would the residual
reduce to the off-block action (H-bd(H))d0. In general it also contains
within-group operator and damping differences; the remaining residual is not
simply information preserved by K-FAC. Measuring against the damped system
includes lambda d0, unlike the undamped residual.
The reduced correction descends in the full residual on U, not necessarily in
any isolated residual component; the safeguard checks the actual proposal.

Saddle-free, damped reduced Newton. The reduced Hessian H^ = U^T H U is k x k
(k <= |S| (2 + 2r)), so it is eigendecomposed exactly once at build time,
H^ = V Lambda V^T, and the per-step solve runs on the saddle-free curvature
(Dauphin et al., 2014) under the same Tikhonov damping lambda as the base step,

        H^sf:= V diag(|lambda_i| + lambda) V^T,
        c*:= -(H^sf)^{-1} U^T r = -V diag(1 / (|lambda_i| + lambda)) V^T U^T r.

A plain damped solve -(H^ + lambda I)^{-1} U^T r on an indefinite H^ follows every
negative-curvature eigendirection *up*, toward the saddle, where the quadratic model
is unbounded below - so the model test alone would accept an arbitrarily long step.
With |lambda_i| the coefficient along every eigendirection has the sign opposite to that
of the residual (descent on the residual), and the damping bounds the step,
||c*|| <= ||U^T r|| / lambda, whatever the spectrum. On a PD H^ the step is the
Tikhonov-damped reduced Newton step -(H^ + lambda I)^{-1} U^T r. The damping is
not a free parameter of the overlay: `damping` is set by the runner to the lambda of
the K-FAC / EKFAC step it corrects - the same value, not the same form of damping: that
step damps its Kronecker factors (K-FAC) or its eigenbasis exactly (EKFAC), this solve
adds lambda to |lambda_i| on the reduced Hessian - so the correction's floor is the
base step's lambda (a floor decoupled from that scale either lets near-flat reduced
directions blow the proposal past the trust region or never acts). When the runner
adapts that damping (`secondorder.damping.Damping`, the Levenberg-Marquardt rule) the
overlay holds the same object and reads it at every use, so the correction follows the
base step's lambda on the step it changes.

Everything is matrix-free: the cross-block action is read off the full HVP by
embed/slice,   H_{vw} z_w = (H * embed_w(z_w))|_v.

Amortized definition. Building U (subspace iteration: 2 n_it r + r HVPs per pair) and
the reduced Hessian H^ = U^T H U (k HVPs) is the expensive part; the method is
therefore defined with a refresh period T shared with the coupling measurement (P_v z
is z zeroed outside block v; hd0 = H d0 is the one HVP the safeguard needs anyway):

  refresh step t0 (t0 mod T == 0), S_{t0} = select(C_{t0}, tau):
    for each (v, w) in S_{t0}:
      (L_vw, R_vw):= top-r singular directions of H_vw        (subspace iteration)
    U:= orth([P_v d0_{t0}, P_w d0_{t0}, P_v L_vw, P_w R_vw  for (v, w) in S_{t0}])
                                                              (k <= |S| (2 + 2r) columns)
    H^:= sym(U^T H U) = V Lambda V^T                          (k HVPs, one k x k eigh)
    cache OverlaySubspace(pairs=S_{t0}, U in factored form, eigenvalues=Lambda, V)

  every step t:
    r_t:= g_t + hd0_t + lambda d0_t
    c*_t:= -V diag(1 / (|lambda_i| + lambda)) V^T U^T r_t
    d1_t:= d0_t + U c*_t
    d_t:= safeguard(d0_t, d1_t)          (exact m(d1) with the fresh H, one HVP)

`build` is the refresh, `correct` the per-step solve on the cached eigenfactors (no
HVP). The safeguard evaluates the fresh damped model on each proposal; for finite
values, acceptance is bounded by its tolerance relative to the base model value.
This is not a guarantee on the loss unless the optional minibatch-loss test is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from hessian.types import LayerID

from .damping import Damping
from .types import HessianOracle, Pair

_RANK_REVEAL_TOL = 1e-8
# Gram eigenvalues below this fraction of the largest are null directions of the raw
# columns (a d0 slice repeated by two pairs sharing a block, d0 aligned with a singular
# direction); above it the Gram - accumulated in float64 - resolves them.
_GRAM_NULL_REL_TOL = 1e-5

# The raw columns C of a subspace, block-sparse: (block slice, d_block x m matrix) per
# pair block; a column lives on one block and is zero elsewhere.
BlockColumns = tuple[tuple[slice, Tensor], ...]


def _embed(values: Tensor, sl: slice, dim: int) -> Tensor:
    """Lift a block-local vector into the full P-dim space (zeros off the block)."""
    col = torch.zeros(dim, device=values.device, dtype=values.dtype)
    col[sl] = values
    return col


def _apply_cols(op: Callable[[Tensor], Tensor], cols: Tensor) -> Tensor:
    """Apply a matrix-free operator column-wise to `cols` (shape d_in x k)."""
    return torch.stack([op(cols[:, j]) for j in range(cols.shape[1])], dim=1)


def _project(columns: BlockColumns, x: Tensor) -> Tensor:
    """C^T x for block-sparse columns: one block-local product per pair block."""
    return torch.cat([a.t() @ x[sl] for sl, a in columns])


def _lift(columns: BlockColumns, c: Tensor, dim: int) -> Tensor:
    """C c, accumulated on the blocks into one full P-dim vector."""
    out = c.new_zeros(dim)
    offset = 0
    for sl, a in columns:
        m = a.shape[1]
        out[sl] += a @ c[offset : offset + m]
        offset += m
    return out


def _orthonormalizing_transform(columns: BlockColumns) -> Tensor | None:
    """T (k x k') with U = C T orthonormal and spanning the column space of C; None if empty.

    Column norms and block products C_i^T C_j are computed in the input dtype,
    then converted to float64 for Gram assembly, norm-based scaling and
    eigendecomposition. Pair blocks on different parameter blocks are exactly
    orthogonal; only those on the same block meet. Zero columns get scale 0,
    hence an empty Gram row. With G = V Lambda V^T the transform is
    T = D^{-1} V_keep Lambda_keep^{-1/2}, keeping the eigenvalues above
    `_GRAM_NULL_REL_TOL` of the largest: the null directions of a rank-deficient
    stack (d0|_v repeated by every pair sharing block v) are dropped exactly, and the
    k x k' factor carries the orthonormalization instead of a materialized P x k'
    basis - 3 GB on ResNet-50, which next to the retained step graph and the HVP's
    transient did not fit a 24 GB card at the protocol batch.
    """
    norms = torch.cat([a.norm(dim=0) for _, a in columns]).double()
    scale = torch.where(norms > _RANK_REVEAL_TOL, 1.0 / norms, torch.zeros_like(norms))
    if not scale.any():
        return None
    k = norms.numel()
    gram = torch.zeros(k, k, dtype=torch.float64, device=norms.device)
    offsets = [0]
    for _, a in columns:
        offsets.append(offsets[-1] + a.shape[1])
    for i, (sl_i, a_i) in enumerate(columns):
        for j, (sl_j, a_j) in enumerate(columns):
            if j < i or sl_j != sl_i:
                continue
            block = (a_i.t() @ a_j).double()
            gram[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block
            gram[offsets[j] : offsets[j + 1], offsets[i] : offsets[i + 1]] = block.t()
    gram = scale[:, None] * gram * scale[None, :]
    lam, vecs = torch.linalg.eigh(gram)
    keep = lam > _GRAM_NULL_REL_TOL * lam[-1]
    if not keep.any():
        return None
    transform = scale[:, None] * vecs[:, keep] / lam[keep].sqrt()
    return transform.to(columns[0][1].dtype)


@dataclass(frozen=True)
class OverlaySubspace:
    """The cached reduced Newton system of the selected pairs, in eigenfactors.

    - ``pairs``: the block pairs S the subspace was built over;
    - ``columns`` / ``transform``: U = C T in factored form - C the raw pair columns
      (d0 slices and singular directions), block-sparse on their parameter blocks,
      and T the k x k' orthonormalizing transform (`_orthonormalizing_transform`), so
      U^T x and U c cost block-local products and no P x k' matrix is ever stored;
    - ``dim``: P, the full parameter dimension U acts on;
    - ``eigenvalues``: Lambda, the k' eigenvalues (ascending) of the reduced Hessian
      H^ = sym(U^T H U) at build time;
    - ``eigenvectors``: V, the matching orthonormal eigenvectors as columns (k' x k').
    """

    pairs: tuple[Pair, ...]
    columns: BlockColumns
    transform: Tensor
    dim: int
    eigenvalues: Tensor
    eigenvectors: Tensor

    def project(self, x: Tensor) -> Tensor:
        """U^T x (k')."""
        return self.transform.t() @ _project(self.columns, x)

    def lift(self, c: Tensor) -> Tensor:
        """U c (P)."""
        return _lift(self.columns, self.transform @ c, self.dim)

    @property
    def basis(self) -> Tensor:
        """U materialized (P x k'), for inspection and tests - not used by the step."""
        k = self.transform.shape[1]
        eye = torch.eye(k, dtype=self.transform.dtype, device=self.transform.device)
        return torch.stack([self.lift(eye[:, j]) for j in range(k)], dim=1)

    @property
    def reduced(self) -> Tensor:
        """H^ = V Lambda V^T, the k' x k' reduced Hessian at build time."""
        v = self.eigenvectors
        return (v * self.eigenvalues) @ v.t()


@dataclass(frozen=True)
class CoupleFacOverlay:
    """Coupling-gated rank-r reduced-Newton correction over a K-FAC direction."""

    rank: int = 2
    n_power_iter: int = 8
    # Tikhonov damping lambda of the reduced solve (|lambda_i| + lambda); the runner
    # passes the base preconditioner's lambda - the same value, not the same form of
    # damping (K-FAC damps its Kronecker factors, EKFAC its eigenbasis) - a float fixes
    # it, a shared `Damping` is read at every use (``float(damping)``).
    damping: float | Damping = 1e-2
    max_pairs: int | None = None

    # -- selection -----------------------------------
    def select(self, coupling: dict[Pair, float], tau: float) -> list[Pair]:
        """S = {(v, w): C(v, w) > tau}, strongest coupling first, capped at max_pairs."""
        flagged = sorted(
            (pair for pair, c in coupling.items() if c > tau),
            key=lambda pair: coupling[pair],
            reverse=True,
        )
        return flagged if self.max_pairs is None else flagged[: self.max_pairs]

    # -- refresh -----------------
    def build(
        self,
        d0: Tensor,
        hvp: HessianOracle,
        block_ranges: dict[LayerID, slice],
        selected: list[Pair],
        *,
        generator: torch.Generator | None = None,
    ) -> OverlaySubspace | None:
        """The joint basis U over ``selected`` and the eigenfactors of H^ = sym(U^T H U).

        Costs, per pair, the subspace iteration (2 n_it r + r HVPs), plus k' HVPs for
        the reduced Hessian on the joint k'-column basis, whose k' x k' eigendecomposition
        is done here once. None when nothing was selected or every column collapsed.
        The basis stays in factored block-sparse form (`OverlaySubspace`): the raw
        columns of a pair occupy only their two blocks, U^T h is taken column by column
        as each HVP arrives, and neither a dense P x k' basis nor H U is materialized -
        with the retained step graph and the HVP's own transient, that is what fits
        ResNet-50 at batch 64 on a 24 GB card (a dense basis did not).
        """
        if not selected:
            return None
        dim = d0.numel()
        columns: list[tuple[slice, Tensor]] = []
        for v, w in selected:
            sv, sw = block_ranges[v], block_ranges[w]
            left, right = self._cross_singular(d0, hvp, sv, sw, generator)
            columns.append((sv, torch.cat([d0[sv, None], left], dim=1)))
            columns.append((sw, torch.cat([d0[sw, None], right], dim=1)))
        raw: BlockColumns = tuple(columns)
        transform = _orthonormalizing_transform(raw)
        if transform is None:
            return None
        k = transform.shape[1]
        eye = torch.eye(k, dtype=transform.dtype, device=transform.device)
        reduced = torch.stack(
            [
                transform.t()
                @ _project(raw, hvp(_lift(raw, transform @ eye[:, j], dim)))
                for j in range(k)
            ],
            dim=1,
        )
        reduced = 0.5 * (reduced + reduced.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
        return OverlaySubspace(
            pairs=tuple(selected),
            columns=raw,
            transform=transform,
            dim=dim,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    # -- correction ---------------
    def correct(
        self, grad: Tensor, d0: Tensor, hd0: Tensor, subspace: OverlaySubspace | None
    ) -> Tensor:
        """d1 = d0 + U c* on the cached eigenfactors (no HVP).

        ``hd0`` is H d0, computed once by the caller (the safeguard reuses it for m(d0)).
        With no subspace d0 itself is returned, untouched.
        """
        if subspace is None:
            return d0
        # Residual of the damped full-Hessian system at the base direction.
        residual = grad + hd0 + float(self.damping) * d0
        return d0 + self._reduced_newton(subspace, residual)

    # -- internals -----------------------------------------------------
    def _cross_singular(
        self,
        ref: Tensor,
        hvp: HessianOracle,
        sv: slice,
        sw: slice,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Top-r singular subspaces of H_{vw} via matrix-free subspace iteration.

        Returns (left in R^{d_v x r}, right in R^{d_w x r}) - the dominant left/right
        singular directions of z_w |-> (H embed_w(z_w))|_v.
        """
        dim = ref.numel()
        d_v, d_w = ref[sv].numel(), ref[sw].numel()
        r = min(self.rank, d_v, d_w)
        if r <= 0:
            return ref.new_zeros(d_v, 0), ref.new_zeros(d_w, 0)

        def matvec(z_w: Tensor) -> Tensor:  # H_{vw}: R^{d_w} -> R^{d_v}
            return hvp(_embed(z_w, sw, dim))[sv]

        def rmatvec(z_v: Tensor) -> Tensor:  # H_{vw}^T: R^{d_v} -> R^{d_w}
            return hvp(_embed(z_v, sv, dim))[sw]

        right = torch.randn(
            d_w, r, device=ref.device, dtype=ref.dtype, generator=generator
        )
        right, _ = torch.linalg.qr(right)
        for _ in range(self.n_power_iter):
            left, _ = torch.linalg.qr(_apply_cols(matvec, right))
            right, _ = torch.linalg.qr(_apply_cols(rmatvec, left))
        left, _ = torch.linalg.qr(_apply_cols(matvec, right))
        return left, right

    def _reduced_newton(self, subspace: OverlaySubspace, residual: Tensor) -> Tensor:
        """Delta = U c*,  c* = -V diag(1 / (|lambda| + damping)) V^T U^T r.

        The saddle-free damped solve on the cached eigenfactors: |lambda| turns every
        negative-curvature direction into descent on the residual, the Tikhonov
        damping bounds ||c*|| by ||U^T r|| / damping.
        """
        lam, vecs = subspace.eigenvalues, subspace.eigenvectors
        inv_curvature = 1.0 / (lam.abs() + float(self.damping))
        c = -(vecs @ (inv_curvature * (vecs.t() @ subspace.project(residual))))
        return subspace.lift(c)
