"""Capacity-fade analysis: per-cycle rollups, fade-model fitting, projections.

Fits simple empirical fade models to a per-cycle capacity series and
selects among them with the small-sample corrected Akaike criterion
(AICc):

=============  ==============================  ================================
name           model                           typical mechanism flavor
=============  ==============================  ================================
linear         ``cap(n) = c0 - k * n``         constant per-cycle loss
exponential    ``cap(n) = c0 * exp(-n / n0)``  proportional (first-order) loss
sqrt           ``cap(n) = c0 - a * sqrt(n)``   diffusion-limited growth
=============  ==============================  ================================

where ``n`` is the cycle number measured from the first observed cycle.
The winning model provides an analytic projection of the cycle count at
which capacity reaches a chosen fraction of the fitted initial capacity
(e.g. cycles to 80%); projections beyond a configurable multiple of the
observed window are flagged as extrapolated.

Fitting refuses (with a typed :class:`InsufficientCyclesError`) when fewer
than :data:`MIN_CYCLES` finite cycles are available — short series cannot
discriminate between the candidate models.

Rollup helpers summarize long-format per-cycle data (capacity retention
and, when a reference capacity is supplied, per-cycle efficiency =
capacity / reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit

__all__ = [
    "MIN_CYCLES",
    "MODEL_NAMES",
    "FadeError",
    "InsufficientCyclesError",
    "FadeFitError",
    "FadeModelFit",
    "FadeProjection",
    "FadeFit",
    "fit_fade",
    "cycle_rollup",
    "fade_summary_table",
]

MIN_CYCLES: int = 4


class FadeError(Exception):
    """Base class for all errors raised by this module."""


class InsufficientCyclesError(FadeError, ValueError):
    """Raised when a capacity series has too few cycles to fit fade models."""


class FadeFitError(FadeError, RuntimeError):
    """Raised when no candidate fade model could be fitted."""


# --------------------------------------------------------------------------- #
# model definitions
# --------------------------------------------------------------------------- #


def _linear(n: npt.NDArray[np.float64], c0: float, k: float) -> npt.NDArray[np.float64]:
    return c0 - k * n


def _exponential(
    n: npt.NDArray[np.float64], c0: float, n0: float
) -> npt.NDArray[np.float64]:
    return c0 * np.exp(-n / n0)


def _sqrt(n: npt.NDArray[np.float64], c0: float, a: float) -> npt.NDArray[np.float64]:
    return c0 - a * np.sqrt(n)


# name -> (function, parameter names, (lower bounds, upper bounds))
_MODELS: dict[
    str,
    tuple[
        Callable[..., npt.NDArray[np.float64]],
        tuple[str, str],
        tuple[list[float], list[float]],
    ],
] = {
    "linear": (_linear, ("c0", "k"), ([0.0, -np.inf], [np.inf, np.inf])),
    "exponential": (_exponential, ("c0", "n0"), ([0.0, 1e-3], [np.inf, 1e9])),
    "sqrt": (_sqrt, ("c0", "a"), ([0.0, -np.inf], [np.inf, np.inf])),
}

MODEL_NAMES: tuple[str, ...] = tuple(_MODELS)


def _aicc_regression(rss: float, n: int, k: int) -> float:
    """AICc for a least-squares fit: ``n ln(rss/n) + 2k + 2k(k+1)/(n-k-1)``.

    A tiny floor on ``rss`` keeps exact fits (rss = 0) ranked best instead
    of producing ``-inf``/``nan``.
    """
    if n <= k + 1:
        return float("inf")
    rss = max(rss, n * 1e-20)
    return float(n * math.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1))


def _initial_guess(
    name: str, n: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> list[float]:
    """Data-driven starting parameters for each fade model."""
    c0 = float(max(y[0], 1e-9))
    n_max = float(max(n[-1], 1.0))
    drop = float(y[0] - y[-1])
    if name == "linear":
        return [c0, drop / n_max if drop != 0.0 else 1e-3]
    if name == "sqrt":
        return [c0, drop / math.sqrt(n_max) if drop != 0.0 else 1e-3]
    # exponential: match the observed endpoint decay when possible
    if y[-1] > 0 and y[0] > 0 and y[-1] < y[0]:
        n0 = n_max / math.log(y[0] / y[-1])
    else:
        n0 = 10.0 * n_max
    return [c0, float(min(max(n0, 1e-3), 1e9))]


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FadeModelFit:
    """One fitted fade model.

    Attributes
    ----------
    name : str
        Model name (``"linear"``, ``"exponential"`` or ``"sqrt"``).
    params : tuple of float
        Fitted parameters, ordered as in ``param_names``.
    param_names : tuple of str
        Parameter names (first is always ``c0``, the fitted initial
        capacity).
    aicc : float
        Small-sample corrected Akaike criterion of the fit.
    rmse : float
        Root-mean-square residual.
    """

    name: str
    params: tuple[float, ...]
    param_names: tuple[str, ...]
    aicc: float
    rmse: float

    def predict(self, cycles: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Predicted capacity at ``cycles`` (measured from the first
        observed cycle)."""
        fn = _MODELS[self.name][0]
        return np.asarray(fn(np.asarray(cycles, dtype=float), *self.params), dtype=float)


@dataclass(frozen=True)
class FadeProjection:
    """Projected cycle count to a capacity-retention target.

    Attributes
    ----------
    model : str
        Model used for the projection.
    fraction : float
        Retention target as a fraction of the fitted initial capacity.
    target_capacity : float
        ``fraction * c0`` in capacity units.
    cycles : float or None
        Projected cycles (from the first observed cycle) at which capacity
        reaches the target; None when the fitted model never reaches it
        (e.g. a non-fading or improving trend).
    extrapolated : bool
        True when the projection lies beyond the extrapolation window
        (or the target is never reached).
    """

    model: str
    fraction: float
    target_capacity: float
    cycles: float | None
    extrapolated: bool


@dataclass(frozen=True)
class FadeFit:
    """Fade-model comparison for one capacity series.

    Attributes
    ----------
    models : dict
        ``{name: FadeModelFit}`` for every model that converged.
    best_model : str
        Name of the AICc-best model.
    n_cycles : int
        Number of finite (cycle, capacity) points used.
    cycle_offset : float
        First observed cycle number; predictions and projections use
        cycles measured from this offset.
    max_cycle : float
        Largest observed cycle, in offset units (i.e. observed window
        length).
    initial_capacity_observed : float
        Capacity at the first observed cycle.
    final_capacity_observed : float
        Capacity at the last observed cycle.
    observed_fade_rate_pct_per_cycle : float
        End-to-end observed fade rate, percent of initial capacity per
        cycle.
    """

    models: dict[str, FadeModelFit]
    best_model: str
    n_cycles: int
    cycle_offset: float
    max_cycle: float
    initial_capacity_observed: float
    final_capacity_observed: float
    observed_fade_rate_pct_per_cycle: float

    @property
    def best(self) -> FadeModelFit:
        """The AICc-best fitted model."""
        return self.models[self.best_model]

    def predict(self, cycles: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Best-model capacity prediction at ``cycles`` (offset units)."""
        return self.best.predict(cycles)

    def cycles_to_fraction(
        self, fraction: float = 0.80, extrapolation_factor: float = 2.0
    ) -> FadeProjection:
        """Project the cycle count at which capacity reaches ``fraction * c0``.

        The AICc-best model is inverted analytically:

        - linear: ``n = c0 (1 - fraction) / k`` (requires ``k > 0``)
        - exponential: ``n = -n0 ln(fraction)``
        - sqrt: ``n = (c0 (1 - fraction) / a)^2`` (requires ``a > 0``)

        where ``c0`` is the *fitted* initial capacity of the best model.

        Parameters
        ----------
        fraction : float, optional
            Retention target in (0, 1). Default 0.80 (cycles to 80%).
        extrapolation_factor : float, optional
            Projections beyond ``extrapolation_factor * max_cycle`` are
            flagged ``extrapolated=True``. Default 2.0.

        Returns
        -------
        FadeProjection
        """
        if not 0.0 < fraction < 1.0:
            raise FadeError(
                f"fraction must be in (0, 1); got {fraction}. Use e.g. 0.80 "
                "for cycles-to-80%-retention."
            )
        if extrapolation_factor <= 0.0:
            raise FadeError(
                f"extrapolation_factor must be positive; got {extrapolation_factor}."
            )
        fit = self.best
        c0 = fit.params[0]
        target = fraction * c0
        cycles: float | None
        if fit.name == "linear":
            k = fit.params[1]
            cycles = c0 * (1.0 - fraction) / k if k > 0.0 else None
        elif fit.name == "exponential":
            n0 = fit.params[1]
            cycles = -n0 * math.log(fraction)
        else:  # sqrt
            a = fit.params[1]
            cycles = (c0 * (1.0 - fraction) / a) ** 2 if a > 0.0 else None
        extrapolated = cycles is None or cycles > extrapolation_factor * self.max_cycle
        return FadeProjection(
            model=fit.name,
            fraction=float(fraction),
            target_capacity=float(target),
            cycles=None if cycles is None else float(cycles),
            extrapolated=bool(extrapolated),
        )


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def fit_fade(
    cycle_index: npt.ArrayLike,
    capacity: npt.ArrayLike,
    min_cycles: int = MIN_CYCLES,
    models: Sequence[str] = MODEL_NAMES,
) -> FadeFit:
    """Fit fade models to a per-cycle capacity series and pick one by AICc.

    Cycle numbers are shifted so the first observed cycle is 0; all
    predictions and projections are in these offset units. Non-finite
    (cycle, capacity) pairs are dropped before fitting.

    Parameters
    ----------
    cycle_index : array_like
        Cycle numbers (need not start at 0 nor be consecutive).
    capacity : array_like
        Capacity at each cycle, same length as ``cycle_index``.
    min_cycles : int, optional
        Minimum number of finite cycles required (default
        :data:`MIN_CYCLES`). Must be at least 4 so that AICc is defined
        for the two-parameter models.
    models : sequence of str, optional
        Subset of :data:`MODEL_NAMES` to fit. Default: all three.

    Returns
    -------
    FadeFit

    Raises
    ------
    InsufficientCyclesError
        If fewer than ``min_cycles`` finite cycles remain — short series
        cannot discriminate between fade models; collect more cycles.
    FadeFitError
        If every candidate model fails to converge.
    FadeError
        On malformed inputs (shape/length mismatch or an unknown model
        name).
    """
    unknown = [m for m in models if m not in _MODELS]
    if unknown:
        raise FadeError(
            f"unknown fade model(s) {unknown}; supported models are {MODEL_NAMES}."
        )
    if int(min_cycles) < 4:
        raise FadeError(
            f"min_cycles must be at least 4 (two-parameter models need "
            f"n > k + 1 = 3 for AICc); got {min_cycles}."
        )
    n_arr = np.asarray(cycle_index, dtype=float)
    y_arr = np.asarray(capacity, dtype=float)
    if n_arr.shape != y_arr.shape or n_arr.ndim != 1:
        raise FadeError(
            "cycle_index and capacity must be one-dimensional arrays of the "
            f"same length; got shapes {n_arr.shape} and {y_arr.shape}."
        )
    mask = np.isfinite(n_arr) & np.isfinite(y_arr)
    n_arr, y_arr = n_arr[mask], y_arr[mask]
    if len(n_arr) < min_cycles:
        raise InsufficientCyclesError(
            f"fade fitting needs at least {min_cycles} finite cycles; got "
            f"{len(n_arr)}. Collect more cycles before fitting fade models."
        )
    order = np.argsort(n_arr, kind="stable")
    n_arr, y_arr = n_arr[order], y_arr[order]
    offset = float(n_arr[0])
    n_arr = n_arr - offset  # first observed cycle = 0

    fitted: dict[str, FadeModelFit] = {}
    errors: dict[str, str] = {}
    for name in models:
        fn, param_names, bounds = _MODELS[name]
        try:
            p0 = _initial_guess(name, n_arr, y_arr)
            popt, _ = curve_fit(fn, n_arr, y_arr, p0=p0, bounds=bounds, maxfev=10000)
            rss = float(np.sum((fn(n_arr, *popt) - y_arr) ** 2))
            fitted[name] = FadeModelFit(
                name=name,
                params=tuple(float(v) for v in popt),
                param_names=param_names,
                aicc=_aicc_regression(rss, len(n_arr), len(popt)),
                rmse=float(math.sqrt(rss / len(n_arr))),
            )
        except (RuntimeError, ValueError) as exc:
            errors[name] = str(exc)
    if not fitted:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise FadeFitError(
            f"no fade model converged on this series ({detail}). Check the "
            "capacity series for pathologies (constant values, sign flips)."
        )
    best = min(fitted, key=lambda k: fitted[k].aicc)
    y0, y_last = float(y_arr[0]), float(y_arr[-1])
    rate = (y0 - y_last) / max(y0, 1e-9) / max(float(n_arr[-1]), 1.0) * 100.0
    return FadeFit(
        models=fitted,
        best_model=best,
        n_cycles=int(len(n_arr)),
        cycle_offset=offset,
        max_cycle=float(n_arr[-1]),
        initial_capacity_observed=y0,
        final_capacity_observed=y_last,
        observed_fade_rate_pct_per_cycle=float(rate),
    )


# --------------------------------------------------------------------------- #
# rollup helpers
# --------------------------------------------------------------------------- #


def cycle_rollup(
    df: pd.DataFrame,
    cycle_col: str = "cycle",
    capacity_col: str = "capacity",
    group_col: str | None = None,
    reference_capacity: float | None = None,
) -> pd.DataFrame:
    """Tidy per-cycle rollup: retention, efficiency and cycle-over-cycle change.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format per-cycle data.
    cycle_col, capacity_col : str, optional
        Column names for the cycle number and capacity.
    group_col : str, optional
        Optional grouping column (one series per group). When omitted the
        whole frame is treated as a single series.
    reference_capacity : float, optional
        Set/nominal capacity in the same units as ``capacity_col``. When
        given, an ``efficiency`` column (capacity / reference) is added —
        the usual per-cycle coulombic-efficiency convention when capacity
        is coulomb-counted against a fixed set capacity.

    Returns
    -------
    pandas.DataFrame
        Sorted by (group,) cycle with columns: (group,) cycle, capacity,
        ``retention`` (capacity / first-cycle capacity of the series),
        ``capacity_change`` (difference vs the previous cycle) and, when a
        reference is supplied, ``efficiency``.

    Raises
    ------
    FadeError
        If required columns are missing or the reference capacity is not
        positive.
    """
    missing = [c for c in (cycle_col, capacity_col) if c not in df.columns]
    if group_col is not None and group_col not in df.columns:
        missing.append(group_col)
    if missing:
        raise FadeError(
            f"cycle_rollup: column(s) {missing} not found in the DataFrame; "
            f"available columns are {list(df.columns)}."
        )
    if reference_capacity is not None and not reference_capacity > 0:
        raise FadeError(
            f"reference_capacity must be positive; got {reference_capacity}."
        )
    keep_cols = ([group_col] if group_col else []) + [cycle_col, capacity_col]
    out = df.loc[:, keep_cols].copy()
    out = out.sort_values(keep_cols[:-1], ignore_index=True)

    def _augment(g: pd.DataFrame) -> pd.DataFrame:
        cap = g[capacity_col].astype(float)
        first = cap.iloc[0]
        g = g.copy()
        g["retention"] = cap / first if first != 0 else np.nan
        g["capacity_change"] = cap.diff()
        if reference_capacity is not None:
            g["efficiency"] = cap / float(reference_capacity)
        return g

    if group_col:
        parts = [_augment(g) for _, g in out.groupby(group_col, sort=False)]
        out = pd.concat(parts, ignore_index=True)
    else:
        out = _augment(out)
    return out


def fade_summary_table(
    df: pd.DataFrame,
    group_col: str,
    cycle_col: str = "cycle",
    capacity_col: str = "capacity",
    reference_capacity: float | None = None,
    min_cycles: int = MIN_CYCLES,
    fraction: float = 0.80,
    extrapolation_factor: float = 2.0,
) -> pd.DataFrame:
    """Per-group fade-fit summary from long-format per-cycle data.

    Fits :func:`fit_fade` to each group's capacity series and tabulates
    the AICc-best model with its retention projection. Groups with fewer
    than ``min_cycles`` finite cycles are skipped (they cannot support a
    model comparison); use :func:`fit_fade` directly if you want the typed
    error instead.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format per-cycle data.
    group_col : str
        Grouping column (one fade fit per group).
    cycle_col, capacity_col : str, optional
        Column names for the cycle number and capacity.
    reference_capacity : float, optional
        Set/nominal capacity; adds first/last ``efficiency`` columns
        (capacity / reference).
    min_cycles : int, optional
        Minimum cycles per group (default :data:`MIN_CYCLES`).
    fraction : float, optional
        Retention target for the projection column (default 0.80).
    extrapolation_factor : float, optional
        Extrapolation flag threshold passed to
        :meth:`FadeFit.cycles_to_fraction`.

    Returns
    -------
    pandas.DataFrame
        One row per fitted group with columns: ``group_col``, ``n_cycles``,
        ``best_model``, ``observed_fade_rate_pct_per_cycle``,
        ``projected_cycles`` (to the retention target), ``extrapolated``
        and, when a reference is supplied, ``efficiency_first`` /
        ``efficiency_last``.
    """
    missing = [c for c in (group_col, cycle_col, capacity_col) if c not in df.columns]
    if missing:
        raise FadeError(
            f"fade_summary_table: column(s) {missing} not found in the "
            f"DataFrame; available columns are {list(df.columns)}."
        )
    rows: list[dict[Hashable, object]] = []
    for key, g in df.sort_values([group_col, cycle_col]).groupby(group_col, sort=False):
        try:
            fit = fit_fade(
                g[cycle_col], g[capacity_col], min_cycles=min_cycles
            )
        except InsufficientCyclesError:
            continue
        proj = fit.cycles_to_fraction(
            fraction=fraction, extrapolation_factor=extrapolation_factor
        )
        row: dict[Hashable, object] = {
            group_col: key,
            "n_cycles": fit.n_cycles,
            "best_model": fit.best_model,
            "observed_fade_rate_pct_per_cycle": round(
                fit.observed_fade_rate_pct_per_cycle, 4
            ),
            "projected_cycles": proj.cycles,
            "extrapolated": proj.extrapolated,
        }
        if reference_capacity is not None:
            cap = g[capacity_col].astype(float).to_numpy()
            finite = cap[np.isfinite(cap)]
            row["efficiency_first"] = float(finite[0] / reference_capacity)
            row["efficiency_last"] = float(finite[-1] / reference_capacity)
        rows.append(row)
    return pd.DataFrame(rows)
