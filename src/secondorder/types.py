"""Shared abstractions for the second-order overlay package (the DIP seam).

`HessianOracle` is the exact full-parameter Hessian-vector product u |-> H u for the
current batch; `Pair` identifies a coupled parameter-block pair (v, w). The overlay and
the safeguard act on flat parameter-space vectors through these abstractions only -
never on a concrete model or a concrete K-FAC library.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from hessian.types import LayerID

# Exact full-parameter Hessian-vector product for the current batch: u |-> H u.
HessianOracle = Callable[[Tensor], Tensor]

# An (ordered) pair of parameter-group identifiers (v, w).
Pair = tuple[LayerID, LayerID]
