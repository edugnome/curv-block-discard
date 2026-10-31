"""Scalar-only reporting for the saved-factor residual geometry diagnostic."""

from __future__ import annotations

import math
from typing import Any

from paper_data.formatting import sci

_ROSTER = {
    (seed, step, draw)
    for seed in (42, 43, 44)
    for step in (20, 60, 200)
    for draw in ("primary", "sensitivity")
}


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"invalid geometry scalar: {name}")
    return float(value)


def _rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = bundle["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != 18
        or {(r["seed"], r["step"], r["drawset"]) for r in rows} != _ROSTER
    ):
        raise ValueError("invalid residual geometry roster")
    for row in rows:
        native, represented = row["native_screen"], row["represented_factor"]
        error = _number(native["max_entry_error"], "native error")
        if error < 0 or native["status"] not in ("resolved", "nonorthogonal_basis"):
            raise ValueError("invalid native screen")
        eta = native["eta_action"]
        if native["status"] == "nonorthogonal_basis" and eta is not None:
            raise ValueError("rejected screen cannot have an action fraction")
        if eta is not None:
            _number(eta, "action fraction")
        if represented["status"] != "available":
            raise ValueError("represented geometry unavailable")
        raw, retained = represented["raw_columns"], represented["retained_columns"]
        if type(raw) is not int or type(retained) is not int or not 0 < retained <= raw:
            raise ValueError("invalid represented geometry dimensions")
        spectral_error = _number(
            represented["gram_spectral_error"], "Gram spectral error"
        )
        low = _number(represented["gram_eigen_min"], "Gram minimum")
        high = _number(represented["gram_eigen_max"], "Gram maximum")
        condition = _number(represented["gram_condition"], "Gram condition")
        difference = _number(represented["map_relative_difference"], "map difference")
        if (
            low <= 0
            or high < low
            or condition < 1
            or difference < 0
            or spectral_error < 0
        ):
            raise ValueError("invalid represented geometry")
    return sorted(rows, key=lambda r: (r["seed"], r["step"], r["drawset"] != "primary"))


def geometry_table(residual_basis_geometry: dict[str, Any]) -> str:
    """Render all 18 rows from compact scalar data only."""
    lines = []
    for row in _rows(residual_basis_geometry):
        n, g = row["native_screen"], row["represented_factor"]
        draw = "P" if row["drawset"] == "primary" else "S"
        screen = "pass" if n["status"] == "resolved" else "fail"
        lines.append(
            f"    {row['seed']} & {row['step']} & {draw} & {g['retained_columns']}/{g['raw_columns']} & ${sci(n['max_entry_error'], 2)}$ & {screen} & ${sci(g['gram_spectral_error'], 2)}$ & {g['gram_eigen_min']:.6f} & {g['gram_eigen_max']:.6f} & {g['gram_condition']:.6f} & {100 * g['map_relative_difference']:.4f} "
            + r"\\"
            + f" % state {row['seed']}/{row['step']}/{row['drawset']}"
        )
    body = "\n".join(lines)
    return rf"""\begin{{table*}}[htbp]
  \centering
  \caption{{Native float32 project/lift max-entry screen and separate float64
    represented-factor geometry for all registered state/draw cases (P/S).
    Gram extrema and condition describe the saved factors; the last column is
    the relative spectral difference between saved and whitened represented maps,
    in percent, not an error bound on native arithmetic or loss.}}
  \label{{tab:residual-geometry}}
  \scriptsize
  \setlength{{\tabcolsep}}{{2pt}}
  \begin{{tabular}}{{@{{}}rrcllllrrrr@{{}}}}
    \toprule
    Seed & Step & Draw & $k$/raw & Native error & Screen & $\|G-I\|_2$ & $\lambda_{{\min}}(G)$ & $\lambda_{{\max}}(G)$ & $\kappa(G)$ & Map diff. (\%) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""


def geometry_numbers(residual_basis_geometry: dict[str, Any]) -> str:
    """Render six numeric macros for the existing residual-ablation resource."""
    rows = _rows(residual_basis_geometry)
    native = [r["native_screen"]["max_entry_error"] for r in rows]
    represented = [r["represented_factor"] for r in rows]
    values = {
        "BasisNativeErrorMin": sci(min(native), 2),
        "BasisNativeErrorMax": sci(max(native), 2),
        "BasisGramEigenMin": f"{min(r['gram_eigen_min'] for r in represented):.6f}",
        "BasisGramEigenMax": f"{max(r['gram_eigen_max'] for r in represented):.6f}",
        "BasisGramConditionMax": f"{max(r['gram_condition'] for r in represented):.6f}",
        "BasisMapDifferencePercentMax": f"{100 * max(r['map_relative_difference'] for r in represented):.3f}",
    }
    return "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items()
    )
