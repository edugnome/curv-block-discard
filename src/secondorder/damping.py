"""
The run's one Tikhonov damping lambda, shared by reference and adapted by the
Levenberg-Marquardt rule (Martens & Grosse, 2015, Section 6.5).

Why one object. The damped local model

  m(d) = g^T d + 1/2 d^T (H + lambda I) d

is used in the overlay residual and safeguard. K-FAC factors use factored damping,
EKFAC uses eigenbasis damping, and the overlay damps the absolute reduced eigenvalues.
A truncated-CG reference uses the exact damped Hessian-vector product; it need not
minimize the model. A single `Damping` is read at provider refresh and at each
overlay and safeguard use.

Optional adaptation applies a Levenberg-Marquardt rule to the actual-to-predicted
reduction of the base step. The damped model includes a quadratic penalty absent from
the loss, so disagreement is not solely evidence of an inaccurate undamped Hessian
approximation. Adaptation does not guarantee model validity.

  rho = (L(theta) - L(theta + d0)) / (-m(d0)),

  m(d0) >= 0  or  rho < 1/4  ->  lambda <- factor * lambda
  rho > 3/4                   ->  lambda <- lambda / factor
  otherwise unchanged;  then clamp to [lambda_min, lambda_max].

Martens & Grosse apply it every T steps with the factor omega_1^{-1} = (20/19)^T
(about 2.8 for T = 20); here the update runs once per payoff record - the records are
T = ``measure_period`` steps apart - with a default factor of 2, the same order and a
little more conservative. The thresholds 1/4 and 3/4 are theirs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The Levenberg-Marquardt acceptance thresholds on rho (Martens & Grosse, 2015).
LM_RHO_LOW = 0.25
LM_RHO_HIGH = 0.75


@dataclass
class Damping:
    """The one mutable Tikhonov damping lambda of a run.

    - ``value``: the lambda currently in force (lambda_0 at construction);
    - ``factor``: the multiplicative LM step (> 1);
    - ``bounds``: (lambda_min, lambda_max) the value is clamped to after every update;
    - ``adapt``: False freezes the value at lambda_0 (`update` still reports rho).

    ``float(damping)`` reads the current value - the one call every consumer makes at
    the moment it needs lambda, so there is exactly one lambda at any time.
    """

    value: float
    factor: float = 2.0
    bounds: tuple[float, float] = (1e-3, 1e3)
    adapt: bool = True

    def __post_init__(self) -> None:
        low, high = self.bounds
        if not 0.0 < low <= high:
            raise ValueError(
                f"damping bounds must satisfy 0 < min <= max, got {self.bounds}"
            )
        if self.factor <= 1.0:
            raise ValueError(f"damping factor must exceed 1, got {self.factor}")

    @classmethod
    def fixed(cls, value: float) -> Damping:
        """A damping that never moves (the fixed-lambda protocol)."""
        return cls(value, adapt=False)

    def __float__(self) -> float:
        return self.value

    def update(self, m0: float, actual: float) -> float:
        """One Levenberg-Marquardt step from the base step's outcome; returns rho.

        ``m0`` = m(d0) is the damped model's predicted change for the step taken,
        ``actual`` = L(theta) - L(theta + d0) the realized decrease on the same
        minibatch; rho = actual / (-m0) (NaN when m0 == 0, where the ratio is
        undefined). With ``adapt`` off the value is left alone and rho is only reported.
        A non-finite rho - a step whose loss overflowed - counts as an over-promise.
        """
        rho = actual / -m0 if m0 != 0.0 else math.nan
        if not self.adapt:
            return rho
        low, high = self.bounds
        if m0 >= 0.0 or math.isnan(rho) or rho < LM_RHO_LOW:
            self.value = min(self.value * self.factor, high)
        elif rho > LM_RHO_HIGH:
            self.value = max(self.value / self.factor, low)
        return rho
