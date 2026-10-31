"""LaTeX number formatting utilities.

Every formatter returns a *raw* LaTeX fragment (no surrounding ``$``);
callers wrap in math mode where needed. Formatting that does not depend on a
bundle's shape lives here; a bundle's own vocabulary and accessors live in
that bundle's ``tables/`` module, which its figure imports.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Missing-value marker
# ---------------------------------------------------------------------------

# A cell with no measurement; distinct from a measured zero.
NA = "---"

# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------


def tex(s: str) -> str:
    """Escape the LaTeX specials that occur in registry keys and model ids."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


# ---------------------------------------------------------------------------
# Percent normalisation
# ---------------------------------------------------------------------------


def as_percent(value: float) -> float:
    """Bundles record accuracy as a fraction in some runners, a percentage in
    others; the reader normalises at the cell."""
    return value * 100 if value <= 1 else value


# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------


def round_str(val: float, decimals: int) -> str:
    """Round *val* and return a fixed-width decimal string.

    >>> round_str(2.5926, 2)
    '2.59'
    """
    return f"{val:.{decimals}f}"


def exact_str(val: float) -> str:
    """Format *val* exactly, with no trailing zeros -- for a quantity that lives on
    a fixed grid rather than one measured to some chosen precision.

    Some macros (exp11's median t* over the validity-radius grid
    ``{1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1}``) are not measurements that need
    a "round to N decimals" convention: the median of grid points is either a grid
    point itself or the exact midpoint of two adjacent ones, so the number is exact,
    not rounded, and a fixed-decimals convention is wrong for it on two counts --
    it fabricates false precision and at three decimals it would print the grid's
    own smallest point, ``1e-4``, as ``0.000``.

    ``%g`` covers this grid without trailing zeros, but this function refuses to
    fall through to scientific notation (e.g. ``1e-05``) silently: that would be a
    sign the value handed in is not actually an exact grid point/midpoint, so a
    caller mistake, not something to paper over -- a ``ValueError`` naming *val* is
    raised instead.

    >>> exact_str(0.175)
    '0.175'
    >>> exact_str(0.03)
    '0.03'
    """
    s = f"{val:g}"
    if "e" in s:
        raise ValueError(f"exact_str: {val!r} formatted in scientific notation ({s})")
    return s


def sign_str(val: float, decimals: int) -> str:
    """Format with explicit LaTeX sign for negative values.

    Positive or zero values render plain; negative values use ``$-$`` prefix
    so that the minus sign is typeset as a math dash.

    >>> sign_str(-0.023, 2)
    '$-$0.02'
    >>> sign_str(0.04, 2)
    '0.04'
    """
    if val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Scientific notation
# ---------------------------------------------------------------------------


def sci(val: float, sig: int = 2) -> str:
    r"""Format *val* in LaTeX scientific notation.

    Returns e.g. ``1.51{\cdot}10^{-2}`` (without surrounding ``$``).
    *sig* is the number of significant digits in the mantissa.

    >>> sci(0.0151, 2)
    '1.51{\\cdot}10^{-2}'
    """
    if val == 0:
        return "0"
    exp = math.floor(math.log10(abs(val)))
    mantissa = val / 10**exp
    # Format mantissa with appropriate decimal places
    dec = sig  # digits after the decimal point in mantissa
    m_str = f"{mantissa:.{dec}f}"
    # Strip trailing zeros only beyond sig digits
    return rf"{m_str}{{\cdot}}10^{{{exp}}}"


# ---------------------------------------------------------------------------
# +/- formatting
# ---------------------------------------------------------------------------


def pm(mean: float, std: float, decimals: int) -> str:
    r"""Format ``mean \pm std`` with fixed decimal places.

    >>> pm(1.65, 0.23, 2)
    '1.65 \\pm 0.23'
    """
    return rf"{mean:.{decimals}f} \pm {std:.{decimals}f}"


def pm_sci(mean: float, std: float, sig: int = 2) -> str:
    r"""Format ``mean \pm std`` using scientific notation for both."""
    return rf"{sci(mean, sig)} \pm {sci(std, sig)}"


# ---------------------------------------------------------------------------
# p-value formatting
# ---------------------------------------------------------------------------


def p_str(p: float) -> str:
    """``p{<}10^{-3}`` or ``p{=}0.012`` (a math-mode fragment)."""
    if not math.isfinite(p):
        return r"p{=}\mathrm{NaN}"
    if p < 1e-3:
        return r"p{<}10^{-3}"
    return f"p{{=}}{p:.3f}"


# ---------------------------------------------------------------------------
# TikZ coordinate helpers
# ---------------------------------------------------------------------------


def _auto_fmt(val: float) -> str:
    """Pick a suitable representation for a tikz coordinate value."""
    av = abs(val)
    if av == 0:
        return "0"
    if av >= 0.1:
        # Use 4 decimal places for values >= 0.1
        return f"{val:.4f}"
    # Scientific notation for small values
    return f"{val:.4e}"


def tikz_coord(x: int | float, y: float, y_err: float | None = None) -> str:
    """Format a single PGFPlots coordinate, optionally with error bars.

    >>> tikz_coord(0, 1.0364e-01, 8.5838e-03)
    '(0,1.0364e-01) +- (0,8.5838e-03)'
    """
    x_s = str(x) if isinstance(x, int) else f"{x:.3f}"
    y_s = _auto_fmt(y)
    if y_err is not None:
        return f"({x_s},{y_s}) +- (0,{_auto_fmt(y_err)})"
    return f"({x_s},{y_s})"


# ---------------------------------------------------------------------------
# Threshold / special formatting
# ---------------------------------------------------------------------------


def less_than(order: int, prefix: str = "") -> str:
    r"""Format as ``{<}\,10^{order}`` or ``{<}\,2{\cdot}10^{order}``.

    >>> less_than(-7)
    '{<}\\,10^{-7}'
    >>> less_than(-6, prefix="2")
    '{<}\\,2{\\cdot}10^{-6}'
    """
    if prefix:
        return rf"{{<}}\,{prefix}{{\cdot}}10^{{{order}}}"
    return rf"{{<}}\,10^{{{order}}}"


def wrap_math(s: str) -> str:
    """Wrap a LaTeX fragment in inline math mode."""
    return f"${s}$"
