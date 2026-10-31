"""`secondorder.damping.Damping`: the Levenberg-Marquardt rule on the shared damping.

Pure-arithmetic checks of the rule the exp7 hooks apply at every payoff record: the
four branches (model not descent, over-promise, trustworthy, in between), the clamps at
both bounds, the frozen (``adapt=False``) protocol that still reports rho, the NaN and
overflow cases, and that ``float(damping)`` is the live value every consumer reads.
"""

from __future__ import annotations

import math

import pytest

from secondorder import Damping
from secondorder.damping import LM_RHO_HIGH, LM_RHO_LOW


def test_thresholds_are_martens_grosse() -> None:
    assert (LM_RHO_LOW, LM_RHO_HIGH) == (0.25, 0.75)


def test_rho_is_actual_over_predicted() -> None:
    """rho = (L(theta) - L(theta + d0)) / (-m(d0)); the value moves after it is read."""
    damping = Damping(1.0)
    assert float(damping) == 1.0
    rho = damping.update(-2.0, 1.0)  # predicted a decrease of 2, got 1
    assert math.isclose(rho, 0.5), "rho = 1 / 2"
    assert float(damping) == 1.0, "0.5 lies in [1/4, 3/4]: lambda unchanged"


def test_four_branches() -> None:
    # (1) the model does not even predict a decrease: damp harder whatever the loss did
    damping = Damping(1.0, factor=2.0)
    rho = damping.update(0.5, 0.3)  # m0 > 0 while the loss actually fell
    assert rho < 0.0 and float(damping) == 2.0, "m(d0) >= 0 -> factor * lambda"
    at_zero = Damping(1.0)  # m0 == 0: rho undefined, still not a descent model
    assert math.isnan(at_zero.update(0.0, 1.0)) and float(at_zero) == 2.0

    # (2) over-promise: rho < 1/4
    damping = Damping(1.0, factor=2.0)
    assert math.isclose(damping.update(-1.0, 0.2), 0.2) and float(damping) == 2.0

    # (3) trustworthy: rho > 3/4
    damping = Damping(1.0, factor=2.0)
    assert math.isclose(damping.update(-1.0, 0.9), 0.9) and float(damping) == 0.5

    # (4) in between (bounds included): unchanged
    for actual in (0.25, 0.5, 0.75):
        damping = Damping(1.0, factor=2.0)
        assert math.isclose(damping.update(-1.0, actual), actual)
        assert float(damping) == 1.0, f"rho = {actual} leaves lambda alone"


def test_factor_is_the_multiplicative_step() -> None:
    damping = Damping(1.0, factor=3.0)
    damping.update(-1.0, 0.0)  # increase
    damping.update(-1.0, 0.0)
    assert math.isclose(float(damping), 9.0)
    damping.update(-1.0, 1.0)  # decrease
    assert math.isclose(float(damping), 3.0)


def test_clamped_to_bounds() -> None:
    damping = Damping(0.9, factor=2.0, bounds=(0.5, 1.5))
    damping.update(-1.0, 0.0)  # 1.8 -> clamped to the upper bound
    assert float(damping) == 1.5
    damping.update(-1.0, 0.0)
    assert float(damping) == 1.5, "stays at the upper bound"
    damping.update(-1.0, 1.0)  # 0.75
    assert float(damping) == 0.75
    damping.update(-1.0, 1.0)  # 0.375 -> clamped to the lower bound
    assert float(damping) == 0.5
    damping.update(-1.0, 1.0)
    assert float(damping) == 0.5, "stays at the lower bound"


def test_frozen_reports_rho_but_never_moves() -> None:
    damping = Damping.fixed(0.1)
    assert damping.adapt is False
    assert math.isclose(damping.update(-1.0, 0.0), 0.0) and float(damping) == 0.1
    assert math.isclose(damping.update(-1.0, 1.0), 1.0) and float(damping) == 0.1
    assert damping.update(0.5, 0.3) < 0.0 and float(damping) == 0.1


def test_non_finite_loss_counts_as_over_promise() -> None:
    """A step whose loss overflowed (actual = -inf or NaN) damps harder, never less."""
    damping = Damping(1.0)
    assert damping.update(-1.0, -math.inf) == -math.inf and float(damping) == 2.0
    damping = Damping(1.0)
    assert math.isnan(damping.update(-1.0, math.nan)) and float(damping) == 2.0


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        Damping(1.0, factor=1.0)
    with pytest.raises(ValueError):
        Damping(1.0, bounds=(1.0, 0.1))
    with pytest.raises(ValueError):
        Damping(1.0, bounds=(0.0, 1.0))


def test_shared_by_reference() -> None:
    """Two holders of one `Damping` read the same value after an update (no copies)."""
    damping = Damping(1.0)
    holders = [damping, damping]
    damping.update(-1.0, 0.0)
    assert [float(h) for h in holders] == [2.0, 2.0]
