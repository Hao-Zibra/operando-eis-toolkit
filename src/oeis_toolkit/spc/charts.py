"""Phase-I statistical process control charts.

Individuals / moving-range (I-MR) control limits, EWMA charts with
configurable smoothing and limit width, and Nelson run rules 1-4 returning
tidy violation records.

Standard constants (subgroup size n = 2 for moving ranges; see any SPC
reference, e.g. Montgomery, *Introduction to Statistical Quality Control*):

- ``d2 = 1.128`` so the short-term sigma estimate is ``MRbar / 1.128``
- ``2.66 = 3 / d2`` so individuals limits are ``mean +/- 2.66 * MRbar``
- ``D4 = 3.267`` so the moving-range chart UCL is ``3.267 * MRbar`` (LCL = 0)

All functions are numpy-pure and return plot-ready containers; plotting
itself is left to the caller. Every entry point validates its inputs and
raises :class:`SPCError` with an actionable message, so the functions are
safe to run unattended in pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "D2",
    "K_INDIVIDUALS",
    "D4",
    "RULE_TEXT",
    "SPCError",
    "IMRLimits",
    "EWMAChart",
    "Violation",
    "imr_limits",
    "ewma",
    "nelson_rules",
    "violations_frame",
]

D2: float = 1.128  # E(MR) / sigma for subgroup size n = 2
K_INDIVIDUALS: float = 2.66  # 3 / d2: individuals-chart limit factor on MRbar
D4: float = 3.267  # moving-range chart UCL factor for n = 2

RULE_TEXT: dict[int, str] = {
    1: "one point beyond 3 sigma of the center line",
    2: "9 consecutive points on the same side of the center line",
    3: "6 consecutive points steadily increasing or decreasing",
    4: "14 consecutive points alternating up and down",
}


class SPCError(ValueError):
    """Raised when input data cannot support the requested chart."""


# --------------------------------------------------------------------------- #
# I-MR limits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IMRLimits:
    """Phase-I individuals + moving-range control limits.

    Attributes
    ----------
    center : float
        Individuals-chart center line (mean of the observations).
    ucl, lcl : float
        Individuals-chart control limits ``center +/- 2.66 * MRbar``.
    moving_range : ndarray
        The ``|x[i+1] - x[i]|`` series (length ``n_points - 1``).
    mr_center : float
        Moving-range chart center line (MRbar).
    mr_ucl, mr_lcl : float
        Moving-range chart limits (``3.267 * MRbar`` and 0).
    sigma : float
        Short-term process sigma estimate ``MRbar / 1.128``.
    n_points : int
        Number of finite observations used.
    """

    center: float
    ucl: float
    lcl: float
    moving_range: npt.NDArray[np.float64]
    mr_center: float
    mr_ucl: float
    mr_lcl: float
    sigma: float
    n_points: int


def imr_limits(values: npt.ArrayLike) -> IMRLimits:
    """Compute Phase-I I-MR control limits from an individuals series.

    Individuals chart: ``CL = mean(x)``, ``UCL/LCL = mean +/- 2.66 * MRbar``.
    Moving-range chart: ``CL = MRbar``, ``UCL = 3.267 * MRbar``, ``LCL = 0``.
    Non-finite values are dropped before computation.

    Parameters
    ----------
    values : array_like
        Individuals observations, in process order.

    Returns
    -------
    IMRLimits

    Raises
    ------
    SPCError
        If fewer than 2 finite points remain, or if every moving range is
        zero (identical values give a zero sigma estimate and undefined
        limits).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise SPCError(
            f"imr_limits needs at least 2 finite points; got {len(x)} after "
            "dropping NaN/inf."
        )
    mr = np.abs(np.diff(x))
    mrbar = float(np.mean(mr))
    if mrbar == 0.0:
        raise SPCError(
            "all moving ranges are zero (the series is constant); the sigma "
            "estimate MRbar / 1.128 is zero and control limits are undefined. "
            "Provide a series with process variation."
        )
    center = float(np.mean(x))
    return IMRLimits(
        center=center,
        ucl=center + K_INDIVIDUALS * mrbar,
        lcl=center - K_INDIVIDUALS * mrbar,
        moving_range=mr,
        mr_center=mrbar,
        mr_ucl=D4 * mrbar,
        mr_lcl=0.0,
        sigma=mrbar / D2,
        n_points=int(len(x)),
    )


# --------------------------------------------------------------------------- #
# EWMA
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EWMAChart:
    """EWMA chart series with time-varying control limits.

    Attributes
    ----------
    z : ndarray
        Smoothed series ``z_i = lam * x_i + (1 - lam) * z_{i-1}`` (a
        non-finite input point carries the previous ``z`` forward).
    center : float
        Center line (the target).
    ucl, lcl : ndarray
        Time-varying limits, same length as the input:
        ``center +/- L * sigma * sqrt(lam/(2-lam) * (1 - (1-lam)^(2i)))``.
    lam : float
        Smoothing constant in (0, 1].
    L : float
        Limit half-width in sigma units.
    sigma : float
        Process sigma used for the limits.
    """

    z: npt.NDArray[np.float64]
    center: float
    ucl: npt.NDArray[np.float64]
    lcl: npt.NDArray[np.float64]
    lam: float
    L: float
    sigma: float

    def violations(self) -> list[int]:
        """Indices where the EWMA statistic falls outside its limits."""
        return [
            int(i)
            for i in range(len(self.z))
            if np.isfinite(self.z[i])
            and (self.z[i] > self.ucl[i] or self.z[i] < self.lcl[i])
        ]


def ewma(
    values: npt.ArrayLike,
    lam: float = 0.2,
    L: float = 3.0,
    target: float | None = None,
    sigma: float | None = None,
) -> EWMAChart:
    """EWMA chart with configurable smoothing (lambda) and limit width (L).

    The recursion is ``z_i = lam * x_i + (1 - lam) * z_{i-1}`` with
    ``z_0`` seeded at the target; the exact time-varying limits are

    ``center +/- L * sigma * sqrt( lam/(2-lam) * (1 - (1-lam)^(2i)) )``.

    Parameters
    ----------
    values : array_like
        Individuals observations, in process order. Non-finite points hold
        the previous EWMA value (the recursion skips them).
    lam : float, optional
        Smoothing constant in (0, 1]; smaller values detect smaller shifts.
        Default 0.2.
    L : float, optional
        Limit half-width in sigma units. Default 3.0.
    target : float, optional
        Center line. Defaults to the mean of the finite observations
        (Phase-I usage).
    sigma : float, optional
        Process sigma for the limits. Defaults to the moving-range
        estimate ``MRbar / 1.128``.

    Returns
    -------
    EWMAChart

    Raises
    ------
    SPCError
        If fewer than 2 finite points are supplied, if ``lam`` is outside
        (0, 1], if ``L <= 0``, or if the defaulted sigma is zero.
    """
    if not 0.0 < lam <= 1.0:
        raise SPCError(f"lam must be in (0, 1]; got {lam}.")
    if L <= 0.0:
        raise SPCError(f"L must be positive; got {L}.")
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) < 2:
        raise SPCError(
            f"ewma needs at least 2 finite points; got {len(finite)} after "
            "dropping NaN/inf."
        )
    if target is None:
        target = float(np.mean(finite))
    if sigma is None:
        sigma = float(np.mean(np.abs(np.diff(finite)))) / D2
    if sigma <= 0.0:
        raise SPCError(
            "sigma must be positive; the moving-range estimate is zero for a "
            "constant series. Pass an explicit sigma or provide a series with "
            "process variation."
        )
    z = np.empty(len(x), dtype=float)
    prev = float(target)
    for i, xi in enumerate(x):
        if np.isfinite(xi):
            prev = lam * xi + (1.0 - lam) * prev
        z[i] = prev
    idx = np.arange(1, len(x) + 1, dtype=float)
    half = (
        L * sigma * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2.0 * idx)))
    )
    return EWMAChart(
        z=z,
        center=float(target),
        ucl=target + half,
        lcl=target - half,
        lam=float(lam),
        L=float(L),
        sigma=float(sigma),
    )


# --------------------------------------------------------------------------- #
# Nelson run rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Violation:
    """One Nelson-rule violation, in tidy-record form.

    Attributes
    ----------
    index : int
        Position in the input series at which the rule condition is met.
        For run rules, every point that completes or extends a qualifying
        run is flagged.
    rule : int
        Nelson rule number (1-4).
    value : float
        The observation at ``index``.
    rule_text : str
        Human-readable rule description (see :data:`RULE_TEXT`).
    """

    index: int
    rule: int
    value: float
    rule_text: str


def nelson_rules(
    values: npt.ArrayLike,
    center: float,
    sigma: float,
    rules: Sequence[int] = (1, 2, 3, 4),
) -> list[Violation]:
    """Evaluate Nelson run rules 1-4 on an individuals series.

    Rules
    -----
    1. One point beyond 3 sigma of the center line.
    2. Nine consecutive points on the same side of the center line.
    3. Six consecutive points steadily increasing or decreasing.
    4. Fourteen consecutive points alternating up and down.

    Points exactly on the center line break rule-2 runs; zero-size steps
    break rule-3/4 runs; non-finite points break every run and are never
    flagged themselves.

    Parameters
    ----------
    values : array_like
        Individuals observations, in process order.
    center : float
        Center line (e.g. ``IMRLimits.center``).
    sigma : float
        Process sigma (e.g. ``IMRLimits.sigma``); must be positive.
    rules : sequence of int, optional
        Which rules to evaluate. Default: all four.

    Returns
    -------
    list of Violation
        Tidy records ``(index, rule, value, rule_text)``, ordered by rule
        then index. Empty when the series is in control.

    Raises
    ------
    SPCError
        If ``sigma <= 0`` or an unknown rule number is requested.
    """
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise SPCError(f"sigma must be a positive finite number; got {sigma}.")
    unknown = [r for r in rules if r not in RULE_TEXT]
    if unknown:
        raise SPCError(
            f"unknown Nelson rule number(s) {unknown}; supported rules are "
            f"{sorted(RULE_TEXT)}."
        )
    x = np.asarray(values, dtype=float)
    n = len(x)
    hits: list[Violation] = []

    def _flag(rule: int, i: int) -> None:
        hits.append(
            Violation(index=int(i), rule=rule, value=float(x[i]), rule_text=RULE_TEXT[rule])
        )

    if 1 in rules:
        for i in range(n):
            if np.isfinite(x[i]) and abs(x[i] - center) > 3.0 * sigma:
                _flag(1, i)

    if 2 in rules:
        run = 0
        prev_side = 0.0
        for i in range(n):
            side = float(np.sign(x[i] - center)) if np.isfinite(x[i]) else 0.0
            if side != 0.0 and side == prev_side:
                run += 1
            else:
                run = 1 if side != 0.0 else 0
            prev_side = side
            if run >= 9:
                _flag(2, i)

    if 3 in rules or 4 in rules:
        d = np.diff(x)
        d_sign = np.where(np.isfinite(d), np.sign(d), 0.0)

    if 3 in rules:
        run = 0
        for i in range(len(d_sign)):
            if d_sign[i] != 0.0 and i > 0 and d_sign[i] == d_sign[i - 1]:
                run += 1
            else:
                run = 1 if d_sign[i] != 0.0 else 0
            if run >= 5:  # 5 same-direction steps = 6 points
                _flag(3, i + 1)

    if 4 in rules:
        run = 0
        for i in range(len(d_sign)):
            if (
                d_sign[i] != 0.0
                and i > 0
                and d_sign[i - 1] != 0.0
                and d_sign[i] == -d_sign[i - 1]
            ):
                run += 1
            else:
                run = 1 if d_sign[i] != 0.0 else 0
            if run >= 13:  # 13 alternating steps = 14 points
                _flag(4, i + 1)

    return hits


def violations_frame(violations: Sequence[Violation]) -> pd.DataFrame:
    """Convert violation records to a tidy DataFrame.

    Parameters
    ----------
    violations : sequence of Violation
        Output of :func:`nelson_rules`.

    Returns
    -------
    pandas.DataFrame
        Columns ``index``, ``rule``, ``value``, ``rule_text`` (empty frame
        with those columns when no violations occurred).
    """
    return pd.DataFrame(
        [
            {
                "index": v.index,
                "rule": v.rule,
                "value": v.value,
                "rule_text": v.rule_text,
            }
            for v in violations
        ],
        columns=["index", "rule", "value", "rule_text"],
    )
