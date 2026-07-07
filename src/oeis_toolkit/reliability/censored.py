r"""Right-censored maximum-likelihood life-data analysis.

This module fits parametric life distributions to right-censored samples,
compares candidate distributions with AICc, and provides B-life quantiles,
profile-likelihood (likelihood-ratio) confidence intervals, the
Kaplan-Meier estimator, and Johnson censoring-adjusted median-rank
plotting positions.

Likelihood construction
-----------------------
Each unit contributes to the censored log-likelihood according to its
event indicator (``event`` = 1 for an observed failure, 0 for a
right-censored unit)::

    ln L(theta) =   sum_{i: event_i = 1}  ln f(t_i; theta)
                  + sum_{i: event_i = 0}  ln S(t_i; theta)

An observed failure contributes the log-density ``ln f`` (the unit failed
*at* ``t_i``); a right-censored unit contributes the log-survival ``ln S``
(the unit was still surviving when observation stopped at ``t_i``, so all
we know is that its life exceeds ``t_i``). The negative log-likelihood is
minimized with the derivative-free Nelder-Mead simplex, except for the
one-parameter exponential distribution which has a closed-form censored
MLE.

Supported distributions and their canonical parameters
------------------------------------------------------
=============  ==========================  ===========================
name           parameters                  meaning
=============  ==========================  ===========================
weibull        ``beta``, ``eta``           shape, scale (characteristic life)
lognormal      ``mu``, ``sigma``           mean / s.d. of ``ln t``
loglogistic    ``mu``, ``sigma``           location / scale of logistic on ``ln t``
normal         ``mu``, ``sigma``           mean / s.d. of ``t``
exponential    ``scale``                   mean life (1 / rate)
=============  ==========================  ===========================

Conventions
-----------
``event``
    1 = failure observed, 0 = right-censored. (Some tools use an inverse
    "censor" flag; convert with ``event = 1 - censor``.)
``B_p``
    The life at which a fraction ``p`` of the population has failed
    (B10 is ``p = 0.10``, B50 is ``p = 0.50``).

All entry points validate their inputs and raise typed exceptions with
actionable messages, so they are safe to call unattended in pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2, norm

__all__ = [
    "DISTRIBUTIONS",
    "ReliabilityError",
    "InvalidDataError",
    "FitConvergenceError",
    "CrossValidationError",
    "CensoredFit",
    "ProfileCI",
    "KaplanMeierResult",
    "EngineEstimate",
    "CrossValidationResult",
    "fit_censored",
    "fit_all_distributions",
    "aicc",
    "aicc_table",
    "quantile",
    "cdf",
    "pdf",
    "sf",
    "hazard",
    "profile_ci",
    "profile_ci_b_life",
    "kaplan_meier",
    "median_ranks",
    "cross_validate_weibull",
]

# --------------------------------------------------------------------------- #
# constants and exceptions
# --------------------------------------------------------------------------- #

DISTRIBUTIONS: tuple[str, ...] = (
    "weibull",
    "lognormal",
    "loglogistic",
    "normal",
    "exponential",
)

_POSITIVE_SUPPORT: frozenset[str] = frozenset(
    {"weibull", "lognormal", "loglogistic", "exponential"}
)

_N_PARAMS: dict[str, int] = {
    "weibull": 2,
    "lognormal": 2,
    "loglogistic": 2,
    "normal": 2,
    "exponential": 1,
}

_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "weibull": ("beta", "eta"),
    "lognormal": ("mu", "sigma"),
    "loglogistic": ("mu", "sigma"),
    "normal": ("mu", "sigma"),
    "exponential": ("scale",),
}


class ReliabilityError(Exception):
    """Base class for all errors raised by this module."""


class InvalidDataError(ReliabilityError, ValueError):
    """Raised when input data cannot support the requested fit."""


class FitConvergenceError(ReliabilityError, RuntimeError):
    """Raised when the numerical optimizer fails to converge."""


class CrossValidationError(ReliabilityError, RuntimeError):
    """Raised when independent fitting engines disagree beyond tolerance."""


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def _validate_sample(
    x: npt.ArrayLike,
    event: npt.ArrayLike,
    distribution: str,
    min_failures: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Coerce and validate a censored sample.

    Parameters
    ----------
    x : array_like
        Lifetimes (failure times for ``event == 1``, censoring times for
        ``event == 0``).
    event : array_like
        Event indicators: 1 = failure observed, 0 = right-censored.
    distribution : str
        Distribution name; positive-support distributions require ``x > 0``.
    min_failures : int
        Minimum number of observed failures required for the fit.

    Returns
    -------
    (ndarray, ndarray)
        Validated float lifetimes and integer event indicators.

    Raises
    ------
    InvalidDataError
        On shape mismatch, non-finite values, invalid event codes,
        non-positive lifetimes (for positive-support distributions), or
        too few observed failures.
    """
    xa = np.asarray(x, dtype=float)
    ea = np.asarray(event)
    if xa.ndim != 1 or ea.ndim != 1:
        raise InvalidDataError(
            f"x and event must be one-dimensional; got shapes {xa.shape} and {ea.shape}."
        )
    if len(xa) != len(ea):
        raise InvalidDataError(
            f"x and event must have the same length; got {len(xa)} and {len(ea)}."
        )
    if len(xa) == 0:
        raise InvalidDataError("empty sample: x and event contain no observations.")
    if not np.all(np.isfinite(xa)):
        bad = int(np.flatnonzero(~np.isfinite(xa))[0])
        raise InvalidDataError(
            f"x contains a non-finite value at index {bad}; drop or impute NaN/inf "
            "before fitting."
        )
    ea_f = np.asarray(ea, dtype=float)
    if not np.all(np.isin(ea_f, (0.0, 1.0))):
        raise InvalidDataError(
            "event must contain only 0 (right-censored) and 1 (failure observed). "
            "If your data uses an inverse censor flag, pass event = 1 - censor."
        )
    ei = ea_f.astype(int)
    if distribution in _POSITIVE_SUPPORT and np.any(xa <= 0):
        bad = int(np.flatnonzero(xa <= 0)[0])
        raise InvalidDataError(
            f"x must be strictly positive for the {distribution} distribution; "
            f"found x[{bad}] = {xa[bad]!r}."
        )
    n_fail = int(ei.sum())
    if n_fail < min_failures:
        raise InvalidDataError(
            f"the {distribution} fit needs at least {min_failures} observed "
            f"failure(s) (event == 1); got {n_fail} of {len(xa)} observations. "
            "A fully (or almost fully) censored sample cannot identify the "
            "parameters."
        )
    return xa, ei


def _check_p(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise InvalidDataError(f"quantile fraction p must be in (0, 1); got {p}.")
    return p


# --------------------------------------------------------------------------- #
# censored log-likelihoods and MLE fitters
# --------------------------------------------------------------------------- #


def _weibull_loglik(
    beta: float,
    eta: float,
    x: npt.NDArray[np.float64],
    event: npt.NDArray[np.int_],
) -> float:
    """Censored Weibull log-likelihood.

    Failures contribute ``ln f = ln(beta/eta) + (beta-1) ln(x/eta) - (x/eta)^beta``;
    censored units contribute ``ln S = -(x/eta)^beta``.
    """
    z = x / eta
    return float(
        np.sum(event * (np.log(beta / eta) + (beta - 1.0) * np.log(z)) - z**beta)
    )


def _fit_weibull(
    x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    lx = np.log(x)
    s = float(np.std(lx))
    # sd(ln T) ~ (pi / sqrt(6)) / beta for a Weibull, so 1.283 / s is a
    # method-of-moments style starting shape.
    beta0 = min(max(1.283 / s, 0.3), 50.0) if s > 0 else 1.0
    eta0 = float(np.median(x))

    def nll(p: npt.NDArray[np.float64]) -> float:
        b, e = p
        if b <= 0.0 or e <= 0.0:
            return np.inf
        return -_weibull_loglik(float(b), float(e), x, event)

    res = minimize(
        nll,
        np.array([beta0, eta0]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )
    _raise_if_failed(res, "weibull")
    beta, eta = (float(v) for v in res.x)
    return {"beta": beta, "eta": eta}, float(-res.fun)


def _fit_lognormal(
    x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    lx = np.log(x)

    def nll(p: npt.NDArray[np.float64]) -> float:
        mu, s = p
        if s <= 0.0:
            return np.inf
        z = (lx - mu) / s
        # failures: ln f(t) = ln phi(z) - ln s - ln t ; censored: ln S = ln(1 - Phi(z))
        ll = np.sum(event * (norm.logpdf(z) - np.log(s) - lx)) + np.sum(
            (1 - event) * norm.logsf(z)
        )
        return float(-ll)

    res = minimize(
        nll,
        np.array([float(np.mean(lx)), float(np.std(lx)) + 1e-3]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )
    _raise_if_failed(res, "lognormal")
    mu, sigma = (float(v) for v in res.x)
    return {"mu": mu, "sigma": sigma}, float(-res.fun)


def _logistic_logpdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Stable standard-logistic log-density (symmetric in z)."""
    az = np.abs(z)
    return -az - 2.0 * np.log1p(np.exp(-az))


def _logistic_logsf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Stable standard-logistic log-survival ln S(z) = -ln(1 + e^z)."""
    return -np.logaddexp(0.0, z)


def _fit_loglogistic(
    x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    lx = np.log(x)

    def nll(p: npt.NDArray[np.float64]) -> float:
        mu, s = p
        if s <= 0.0:
            return np.inf
        z = (lx - mu) / s
        # failures: ln f(t) = ln f_logistic(z) - ln s - ln t ; censored: ln S(z)
        ll = np.sum(event * (_logistic_logpdf(z) - np.log(s) - lx)) + np.sum(
            (1 - event) * _logistic_logsf(z)
        )
        return float(-ll)

    res = minimize(
        nll,
        np.array([float(np.mean(lx)), float(np.std(lx)) + 1e-3]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )
    _raise_if_failed(res, "loglogistic")
    mu, sigma = (float(v) for v in res.x)
    return {"mu": mu, "sigma": sigma}, float(-res.fun)


def _fit_normal(
    x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    def nll(p: npt.NDArray[np.float64]) -> float:
        mu, s = p
        if s <= 0.0:
            return np.inf
        z = (x - mu) / s
        # failures: ln f = ln phi(z) - ln s ; censored: ln S = ln(1 - Phi(z))
        ll = np.sum(event * (norm.logpdf(z) - np.log(s))) + np.sum(
            (1 - event) * norm.logsf(z)
        )
        return float(-ll)

    res = minimize(
        nll,
        np.array([float(np.mean(x)), float(np.std(x)) + 1e-3]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )
    _raise_if_failed(res, "normal")
    mu, sigma = (float(v) for v in res.x)
    return {"mu": mu, "sigma": sigma}, float(-res.fun)


def _fit_exponential(
    x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    """Closed-form censored exponential MLE: scale = total time / failures."""
    r = int(event.sum())
    scale = float(x.sum() / r)
    # failures contribute ln f = -ln(scale) - t/scale; censored contribute -t/scale
    ll = float(-r * math.log(scale) - x.sum() / scale)
    return {"scale": scale}, ll


def _raise_if_failed(res, distribution: str) -> None:
    if not res.success or not np.isfinite(res.fun):
        raise FitConvergenceError(
            f"Nelder-Mead did not converge for the {distribution} fit "
            f"({res.message}). Check for degenerate data (e.g. all lifetimes "
            "identical) or rescale the lifetimes to order 1-100 and retry."
        )


_FITTERS: dict[
    str,
    Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.int_]],
        tuple[dict[str, float], float],
    ],
] = {
    "weibull": _fit_weibull,
    "lognormal": _fit_lognormal,
    "loglogistic": _fit_loglogistic,
    "normal": _fit_normal,
    "exponential": _fit_exponential,
}


# --------------------------------------------------------------------------- #
# distribution functions (CDF / PDF / SF / hazard / quantile)
# --------------------------------------------------------------------------- #


def _param_tuple(distribution: str, params: Mapping[str, float]) -> tuple[float, ...]:
    names = _PARAM_NAMES.get(distribution)
    if names is None:
        raise InvalidDataError(
            f"unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}."
        )
    try:
        return tuple(float(params[k]) for k in names)
    except KeyError as exc:
        raise InvalidDataError(
            f"params for {distribution!r} must provide keys {names}; got "
            f"{sorted(params)}."
        ) from exc


def cdf(
    q: npt.ArrayLike, distribution: str, params: Mapping[str, float]
) -> npt.NDArray[np.float64]:
    """Cumulative distribution function F(q) for a supported distribution.

    Parameters
    ----------
    q : array_like
        Evaluation points. For positive-support distributions, values
        ``q <= 0`` return 0.
    distribution : str
        One of :data:`DISTRIBUTIONS`.
    params : mapping
        Canonical parameters (see module docstring).

    Returns
    -------
    ndarray
        F(q), same shape as ``q``.
    """
    qa = np.asarray(q, dtype=float)
    p = _param_tuple(distribution, params)
    if distribution == "normal":
        mu, sigma = p
        return np.asarray(norm.cdf((qa - mu) / sigma), dtype=float)
    out = np.zeros_like(qa, dtype=float)
    pos = qa > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "weibull":
            beta, eta = p
            out[pos] = 1.0 - np.exp(-((qa[pos] / eta) ** beta))
        elif distribution == "lognormal":
            mu, sigma = p
            out[pos] = norm.cdf((np.log(qa[pos]) - mu) / sigma)
        elif distribution == "loglogistic":
            mu, sigma = p
            z = (np.log(qa[pos]) - mu) / sigma
            out[pos] = 1.0 / (1.0 + np.exp(-z))
        elif distribution == "exponential":
            (scale,) = p
            out[pos] = 1.0 - np.exp(-qa[pos] / scale)
    return out


def pdf(
    q: npt.ArrayLike, distribution: str, params: Mapping[str, float]
) -> npt.NDArray[np.float64]:
    """Probability density function f(q) for a supported distribution."""
    qa = np.asarray(q, dtype=float)
    p = _param_tuple(distribution, params)
    if distribution == "normal":
        mu, sigma = p
        return np.asarray(norm.pdf((qa - mu) / sigma) / sigma, dtype=float)
    out = np.zeros_like(qa, dtype=float)
    pos = qa > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "weibull":
            beta, eta = p
            z = qa[pos] / eta
            out[pos] = (beta / eta) * z ** (beta - 1.0) * np.exp(-(z**beta))
        elif distribution == "lognormal":
            mu, sigma = p
            z = (np.log(qa[pos]) - mu) / sigma
            out[pos] = norm.pdf(z) / (sigma * qa[pos])
        elif distribution == "loglogistic":
            mu, sigma = p
            z = (np.log(qa[pos]) - mu) / sigma
            out[pos] = np.exp(_logistic_logpdf(z)) / (sigma * qa[pos])
        elif distribution == "exponential":
            (scale,) = p
            out[pos] = np.exp(-qa[pos] / scale) / scale
    return out


def sf(
    q: npt.ArrayLike, distribution: str, params: Mapping[str, float]
) -> npt.NDArray[np.float64]:
    """Survival function S(q) = 1 - F(q)."""
    return 1.0 - cdf(q, distribution, params)


def hazard(
    q: npt.ArrayLike, distribution: str, params: Mapping[str, float]
) -> npt.NDArray[np.float64]:
    """Hazard function h(q) = f(q) / S(q).

    For the Weibull this is ``(beta/eta) (q/eta)^(beta-1)``: ``beta > 1``
    means a strictly increasing hazard (wear-out), ``beta = 1`` a constant
    hazard (memoryless), ``beta < 1`` a decreasing hazard (infant
    mortality). Points where S(q) underflows to 0 return ``inf``.
    """
    qa = np.asarray(q, dtype=float)
    f = pdf(qa, distribution, params)
    s = sf(qa, distribution, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(s > 0.0, f / np.where(s > 0.0, s, 1.0), np.inf)
    return np.asarray(h, dtype=float)


def quantile(distribution: str, params: Mapping[str, float], p: float) -> float:
    """Life quantile (B-life): the time by which a fraction ``p`` has failed.

    Parameters
    ----------
    distribution : str
        One of :data:`DISTRIBUTIONS`.
    params : mapping
        Canonical parameters (see module docstring).
    p : float
        Failed fraction, in (0, 1). ``p = 0.10`` gives B10, ``p = 0.50`` B50.

    Returns
    -------
    float
        The B-life.
    """
    p = _check_p(p)
    v = _param_tuple(distribution, params)
    if distribution == "weibull":
        beta, eta = v
        return float(eta * (-math.log1p(-p)) ** (1.0 / beta))
    if distribution == "lognormal":
        mu, sigma = v
        return float(math.exp(mu + norm.ppf(p) * sigma))
    if distribution == "loglogistic":
        mu, sigma = v
        return float(math.exp(mu + sigma * math.log(p / (1.0 - p))))
    if distribution == "normal":
        mu, sigma = v
        return float(mu + norm.ppf(p) * sigma)
    # exponential
    (scale,) = v
    return float(-math.log1p(-p) * scale)


# --------------------------------------------------------------------------- #
# fit container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CensoredFit:
    """Result of a censored maximum-likelihood fit.

    Attributes
    ----------
    distribution : str
        Distribution name.
    params : dict
        Canonical parameters (see module docstring).
    log_likelihood : float
        Maximized censored log-likelihood.
    n_total : int
        Number of observations in the sample.
    n_failures : int
        Number of observed failures (``event == 1``).
    """

    distribution: str
    params: dict[str, float]
    log_likelihood: float
    n_total: int
    n_failures: int

    @property
    def n_censored(self) -> int:
        """Number of right-censored observations."""
        return self.n_total - self.n_failures

    @property
    def n_parameters(self) -> int:
        """Number of free parameters in the distribution."""
        return _N_PARAMS[self.distribution]

    @property
    def aicc(self) -> float:
        """Small-sample corrected Akaike information criterion."""
        return aicc(self.log_likelihood, self.n_parameters, self.n_total)

    def b_life(self, p: float = 0.10) -> float:
        """B-life quantile: time by which a fraction ``p`` has failed."""
        return quantile(self.distribution, self.params, p)

    def cdf(self, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Fitted cumulative distribution function F(q)."""
        return cdf(q, self.distribution, self.params)

    def pdf(self, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Fitted probability density function f(q)."""
        return pdf(q, self.distribution, self.params)

    def sf(self, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Fitted survival function S(q)."""
        return sf(q, self.distribution, self.params)

    def hazard(self, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Fitted hazard function h(q) = f(q)/S(q)."""
        return hazard(q, self.distribution, self.params)


def fit_censored(
    x: npt.ArrayLike, event: npt.ArrayLike, distribution: str = "weibull"
) -> CensoredFit:
    """Fit one distribution to a right-censored sample by maximum likelihood.

    The censored log-likelihood sums ``ln f(t_i)`` over observed failures
    and ``ln S(t_i)`` over right-censored units (see module docstring), and
    is maximized with Nelder-Mead (closed form for the exponential).

    Parameters
    ----------
    x : array_like
        Lifetimes: failure times where ``event == 1``, censoring times
        where ``event == 0``. Must be strictly positive except for the
        normal distribution.
    event : array_like
        Event indicators: 1 = failure observed, 0 = right-censored.
    distribution : str, optional
        One of :data:`DISTRIBUTIONS`. Default ``"weibull"``.

    Returns
    -------
    CensoredFit

    Raises
    ------
    InvalidDataError
        If inputs are malformed or contain too few observed failures
        (two-parameter distributions need at least 2; the exponential
        needs at least 1).
    FitConvergenceError
        If the optimizer fails.
    """
    if distribution not in DISTRIBUTIONS:
        raise InvalidDataError(
            f"unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}."
        )
    min_fail = 1 if distribution == "exponential" else 2
    xa, ev = _validate_sample(x, event, distribution, min_failures=min_fail)
    params, ll = _FITTERS[distribution](xa, ev)
    return CensoredFit(
        distribution=distribution,
        params=params,
        log_likelihood=ll,
        n_total=len(xa),
        n_failures=int(ev.sum()),
    )


def fit_all_distributions(
    x: npt.ArrayLike,
    event: npt.ArrayLike,
    distributions: Sequence[str] = DISTRIBUTIONS,
) -> dict[str, CensoredFit]:
    """Fit every requested distribution to the same censored sample.

    Parameters
    ----------
    x, event : array_like
        Censored sample (see :func:`fit_censored`).
    distributions : sequence of str, optional
        Subset of :data:`DISTRIBUTIONS` to fit. Default: all five.

    Returns
    -------
    dict
        ``{distribution: CensoredFit}`` for each requested distribution.
    """
    return {d: fit_censored(x, event, d) for d in distributions}


# --------------------------------------------------------------------------- #
# AICc model comparison
# --------------------------------------------------------------------------- #


def aicc(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Small-sample corrected Akaike information criterion.

    ``AICc = -2 ln L + 2k + 2k(k+1) / (n - k - 1)``.

    Raises
    ------
    InvalidDataError
        If ``n_observations <= n_parameters + 1`` (the correction term is
        undefined); collect more observations or fit a simpler model.
    """
    k, n = int(n_parameters), int(n_observations)
    if n <= k + 1:
        raise InvalidDataError(
            f"AICc is undefined for n = {n} observations and k = {k} parameters "
            "(needs n > k + 1); collect more observations or use a simpler model."
        )
    return float(-2.0 * log_likelihood + 2.0 * k + (2.0 * k * (k + 1)) / (n - k - 1))


def aicc_table(
    x: npt.ArrayLike,
    event: npt.ArrayLike,
    distributions: Sequence[str] = DISTRIBUTIONS,
    b_lives: Sequence[float] = (0.10, 0.50),
) -> pd.DataFrame:
    """AICc comparison table across candidate distributions.

    Fits each distribution by censored MLE and tabulates fit quality and
    headline B-lives, sorted best (lowest AICc) first.

    Parameters
    ----------
    x, event : array_like
        Censored sample (see :func:`fit_censored`).
    distributions : sequence of str, optional
        Distributions to compare. Default: all five.
    b_lives : sequence of float, optional
        Failed fractions to report as B-life columns (default B10 and B50).

    Returns
    -------
    pandas.DataFrame
        Columns: ``distribution``, ``shape``, ``scale``, ``log_likelihood``,
        ``n_parameters``, ``aicc``, ``delta_aicc`` and one ``b<percent>``
        column per requested fraction. ``shape``/``scale`` follow the usual
        reporting convention: weibull (beta, eta); lognormal and
        loglogistic (sigma, median = exp(mu)); normal (sigma, mu);
        exponential (NaN, mean).
    """
    fits = fit_all_distributions(x, event, distributions)
    rows: list[dict[str, object]] = []
    for name, f in fits.items():
        if name == "weibull":
            shape, scale = f.params["beta"], f.params["eta"]
        elif name in ("lognormal", "loglogistic"):
            shape, scale = f.params["sigma"], math.exp(f.params["mu"])
        elif name == "normal":
            shape, scale = f.params["sigma"], f.params["mu"]
        else:  # exponential
            shape, scale = float("nan"), f.params["scale"]
        row: dict[str, object] = {
            "distribution": name,
            "shape": float(shape),
            "scale": float(scale),
            "log_likelihood": f.log_likelihood,
            "n_parameters": f.n_parameters,
            "aicc": f.aicc,
        }
        for p in b_lives:
            row[f"b{round(float(p) * 100):d}"] = f.b_life(p)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("aicc", ignore_index=True)
    table.insert(
        table.columns.get_loc("aicc") + 1,
        "delta_aicc",
        table["aicc"] - table["aicc"].min(),
    )
    return table


# --------------------------------------------------------------------------- #
# profile-likelihood confidence intervals (Weibull)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProfileCI:
    """Profile-likelihood confidence interval.

    Attributes
    ----------
    parameter : str
        Which quantity the interval covers (``"beta"``, ``"eta"``, or a
        B-life label such as ``"B10"``).
    estimate : float
        Maximum-likelihood estimate.
    lower, upper : float
        Interval endpoints; NaN when a bound was not reached within the
        search range (e.g. an unbounded upper limit under heavy censoring).
    level : float
        Confidence level (e.g. 0.95).
    """

    parameter: str
    estimate: float
    lower: float
    upper: float
    level: float


def _lr_bound(
    deviance: Callable[[float], float],
    mle: float,
    threshold: float,
    direction: int,
    factor: float = 1.15,
    max_steps: int = 200,
) -> float:
    """Find one likelihood-ratio interval endpoint by bracketing + brentq.

    Walks multiplicatively away from the MLE until the deviance
    ``2 (lnL_max - lnL_profile)`` exceeds ``threshold``, then root-finds
    the crossing. Returns NaN if no crossing is found within the search
    range (the bound is effectively unbounded on that side).
    """

    def g(v: float) -> float:
        d = deviance(v)
        # a non-finite deviance means the value is far outside the
        # plausible region; treat it as a large finite exceedance so that
        # brentq (which needs finite endpoints) can bracket the crossing.
        return d - threshold if np.isfinite(d) else 1e12

    step = factor if direction > 0 else 1.0 / factor
    v_in = mle
    for _ in range(max_steps):
        v_out = v_in * step
        if g(v_out) > 0.0:
            return float(
                brentq(g, min(v_in, v_out), max(v_in, v_out), xtol=1e-12, rtol=1e-8)
            )
        v_in = v_out
    return float("nan")


def _weibull_profile_setup(
    x: npt.ArrayLike, event: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], float, float, float]:
    xa, ev = _validate_sample(x, event, "weibull", min_failures=2)
    fit = _fit_weibull(xa, ev)
    params, lmax = fit
    return xa, ev, params["beta"], params["eta"], lmax


def _profiled_eta_given_beta(
    beta: float, x: npt.NDArray[np.float64], event: npt.NDArray[np.int_]
) -> float:
    """Closed-form profile scale: for fixed shape, eta^beta = sum(x^beta) / r.

    Computed in log space to stay finite for large shapes.
    """
    r = float(event.sum())
    return float(math.exp((logsumexp(beta * np.log(x)) - math.log(r)) / beta))


def profile_ci(
    x: npt.ArrayLike,
    event: npt.ArrayLike,
    parameter: str = "beta",
    level: float = 0.95,
) -> ProfileCI:
    """Profile-likelihood CI for the Weibull shape or scale.

    Inverts the likelihood-ratio test: a value ``v`` is inside the interval
    when ``2 (lnL_max - lnL_profile(v)) <= chi2(1, level)``, where the
    profile log-likelihood maximizes the censored log-likelihood over the
    other parameter (closed form over the scale; 1-D bounded optimization
    over the shape).

    Parameters
    ----------
    x, event : array_like
        Censored sample (see :func:`fit_censored`).
    parameter : {"beta", "eta"}
        Which Weibull parameter to profile.
    level : float, optional
        Confidence level in (0, 1). Default 0.95.

    Returns
    -------
    ProfileCI
        Endpoints are NaN when unbounded within the search range.
    """
    if parameter not in ("beta", "eta"):
        raise InvalidDataError(
            f"parameter must be 'beta' or 'eta'; got {parameter!r}. "
            "For B-life intervals use profile_ci_b_life()."
        )
    if not 0.0 < level < 1.0:
        raise InvalidDataError(f"level must be in (0, 1); got {level}.")
    xa, ev, bh, eh, lmax = _weibull_profile_setup(x, event)
    threshold = float(chi2.ppf(level, 1))

    if parameter == "beta":

        def deviance(b: float) -> float:
            if b <= 0.0:
                return np.inf
            eta_b = _profiled_eta_given_beta(b, xa, ev)
            return 2.0 * (lmax - _weibull_loglik(b, eta_b, xa, ev))

        mle = bh
    else:
        k_lo, k_hi = max(1e-3, bh / 100.0), bh * 100.0

        def deviance(e: float) -> float:
            if e <= 0.0:
                return np.inf
            res = minimize_scalar(
                lambda b: -_weibull_loglik(b, e, xa, ev),
                bounds=(k_lo, k_hi),
                method="bounded",
            )
            return 2.0 * (lmax + res.fun)

        mle = eh

    lower = _lr_bound(deviance, mle, threshold, direction=-1)
    upper = _lr_bound(deviance, mle, threshold, direction=+1)
    return ProfileCI(
        parameter=parameter, estimate=mle, lower=lower, upper=upper, level=level
    )


def profile_ci_b_life(
    x: npt.ArrayLike,
    event: npt.ArrayLike,
    p: float = 0.10,
    level: float = 0.95,
) -> ProfileCI:
    """Profile-likelihood CI for the Weibull B-life at an arbitrary fraction.

    Reparameterization: fixing ``B_p = v`` pins the scale to
    ``eta = v / (-ln(1 - p))^(1/beta)``, so the profile maximizes the
    censored log-likelihood over the shape alone. The interval is the set
    of ``v`` with ``2 (lnL_max - lnL_profile(v)) <= chi2(1, level)``.

    Parameters
    ----------
    x, event : array_like
        Censored sample (see :func:`fit_censored`).
    p : float, optional
        Failed fraction of the B-life (default 0.10 for B10).
    level : float, optional
        Confidence level in (0, 1). Default 0.95.

    Returns
    -------
    ProfileCI
        Endpoints are NaN when unbounded within the search range (common
        for the upper bound under heavy censoring).
    """
    p = _check_p(p)
    if not 0.0 < level < 1.0:
        raise InvalidDataError(f"level must be in (0, 1); got {level}.")
    xa, ev, bh, eh, lmax = _weibull_profile_setup(x, event)
    threshold = float(chi2.ppf(level, 1))
    c = -math.log1p(-p)
    bp_hat = float(eh * c ** (1.0 / bh))
    k_lo, k_hi = max(1e-3, bh / 100.0), bh * 100.0

    def deviance(v: float) -> float:
        if v <= 0.0:
            return np.inf

        def nll(b: float) -> float:
            eta_v = v / c ** (1.0 / b)
            return -_weibull_loglik(b, eta_v, xa, ev)

        res = minimize_scalar(nll, bounds=(k_lo, k_hi), method="bounded")
        return 2.0 * (lmax + res.fun)

    lower = _lr_bound(deviance, bp_hat, threshold, direction=-1)
    upper = _lr_bound(deviance, bp_hat, threshold, direction=+1)
    pct = p * 100.0
    label = (
        f"B{round(pct):d}"
        if math.isclose(pct, round(pct), abs_tol=1e-9)
        else f"B(p={p:g})"
    )
    return ProfileCI(
        parameter=label, estimate=bp_hat, lower=lower, upper=upper, level=level
    )


# --------------------------------------------------------------------------- #
# nonparametric estimators
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KaplanMeierResult:
    """Kaplan-Meier survival estimate as a right-continuous step function.

    Attributes
    ----------
    times : ndarray
        Step-change times, starting at 0.
    survival : ndarray
        Survival probability immediately after each time in ``times``;
        starts at 1.0.
    """

    times: npt.NDArray[np.float64]
    survival: npt.NDArray[np.float64]

    def survival_at(self, t: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """Evaluate the step function S(t) at one or more times ``t >= 0``.

        Returns a float for scalar input, an ndarray for array input.
        """
        ta = np.asarray(t, dtype=float)
        scalar = ta.ndim == 0
        ta = np.atleast_1d(ta)
        idx = np.searchsorted(self.times, ta, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        out = self.survival[idx]
        return float(out[0]) if scalar else out


def kaplan_meier(x: npt.ArrayLike, event: npt.ArrayLike) -> KaplanMeierResult:
    """Kaplan-Meier (product-limit) estimator for right-censored data.

    At each distinct failure time ``t`` the survival multiplies by
    ``1 - d_t / n_t`` where ``d_t`` is the number of failures at ``t`` and
    ``n_t`` the number at risk (units with lifetime >= t). Censored-only
    times reduce the risk set but add no step.

    Parameters
    ----------
    x : array_like
        Lifetimes (positive).
    event : array_like
        1 = failure observed, 0 = right-censored.

    Returns
    -------
    KaplanMeierResult
        ``times``/``survival`` arrays (prefixed with (0, 1.0)) suitable for
        a post-style step plot, with a :meth:`~KaplanMeierResult.survival_at`
        evaluator.
    """
    xa, ev = _validate_sample(x, event, "weibull", min_failures=1)
    order = np.argsort(xa, kind="stable")
    xs, es = xa[order], ev[order]
    n = len(xs)
    t_out: list[float] = [0.0]
    s_out: list[float] = [1.0]
    surv = 1.0
    for t in np.unique(xs):
        at_risk = int(np.sum(xs >= t))
        d = int(np.sum((xs == t) & (es == 1)))
        if d > 0:
            surv *= 1.0 - d / at_risk
            t_out.append(float(t))
            s_out.append(surv)
    return KaplanMeierResult(
        times=np.asarray(t_out, dtype=float), survival=np.asarray(s_out, dtype=float)
    )


def median_ranks(
    x: npt.ArrayLike, event: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Johnson censoring-adjusted median-rank plotting positions.

    For each observed failure (in ascending-lifetime order) the adjusted
    rank increases by ``(n + 1 - previous adjusted rank) / (1 + number of
    units remaining at and beyond this position)``, and the plotting
    position is the Bernard approximation
    ``F = (adjusted rank - 0.3) / (n + 0.4)``.

    These are diagnostic plotting positions only; fitted lines should come
    from the censored MLE (:func:`fit_censored`), not from regression on
    these points.

    Parameters
    ----------
    x : array_like
        Lifetimes (failures and censoring times mixed).
    event : array_like
        1 = failure observed, 0 = right-censored.

    Returns
    -------
    (ndarray, ndarray)
        Failure lifetimes in ascending order and their plotting positions
        ``F``. Only failures get a point; censored units shift subsequent
        rank increments. At least one observed failure is required.
    """
    xa, ev = _validate_sample(x, event, "weibull", min_failures=1)
    n = len(xa)
    order = np.argsort(xa, kind="stable")
    xs, es = xa[order], ev[order]
    prev = 0.0
    times: list[float] = []
    ranks: list[float] = []
    for i in range(n):
        if es[i] == 1:
            increment = (n + 1 - prev) / (1 + (n - i))
            adjusted = prev + increment
            prev = adjusted
            times.append(float(xs[i]))
            ranks.append((adjusted - 0.3) / (n + 0.4))
    return np.asarray(times, dtype=float), np.asarray(ranks, dtype=float)


# --------------------------------------------------------------------------- #
# optional cross-engine validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineEstimate:
    """Weibull estimate from one fitting engine."""

    beta: float
    eta: float
    b10: float


@dataclass(frozen=True)
class CrossValidationResult:
    """Outcome of a multi-engine Weibull cross-check.

    Attributes
    ----------
    estimates : dict
        ``{engine_name: EngineEstimate}`` for every engine that ran.
    skipped : dict
        ``{engine_name: reason}`` for engines that were not importable.
    tolerance : float
        Maximum allowed pairwise relative difference.
    max_relative_difference : float
        Largest pairwise relative difference actually observed (0.0 when
        only one engine ran).
    """

    estimates: dict[str, EngineEstimate]
    skipped: dict[str, str] = field(default_factory=dict)
    tolerance: float = 0.01
    max_relative_difference: float = 0.0


def cross_validate_weibull(
    x: npt.ArrayLike, event: npt.ArrayLike, tol: float = 0.01
) -> CrossValidationResult:
    """Cross-check the censored Weibull fit against independent engines.

    Always runs this module's MLE (engine ``"oeis-mle"``). Additionally
    fits with ``lifelines`` (``WeibullFitter``: ``lambda_`` = eta,
    ``rho_`` = beta) and the ``reliability`` package
    (``Fit_Weibull_2P``: ``alpha`` = eta, ``beta`` = beta) **if those
    packages are importable** — they are optional and never a hard
    dependency. Engines that cannot be imported are recorded in
    ``skipped`` with an installation hint instead of raising.

    Parameters
    ----------
    x, event : array_like
        Censored sample (see :func:`fit_censored`).
    tol : float, optional
        Maximum allowed pairwise relative difference on beta, eta and B10
        (default 0.01, i.e. 1%).

    Returns
    -------
    CrossValidationResult

    Raises
    ------
    CrossValidationError
        If any two engines disagree on beta, eta or B10 by more than
        ``tol`` (relative).
    """
    xa, ev = _validate_sample(x, event, "weibull", min_failures=2)
    fit = fit_censored(xa, ev, "weibull")
    estimates: dict[str, EngineEstimate] = {
        "oeis-mle": EngineEstimate(
            beta=fit.params["beta"], eta=fit.params["eta"], b10=fit.b_life(0.10)
        )
    }
    skipped: dict[str, str] = {}

    try:
        from lifelines import WeibullFitter  # type: ignore[import-not-found]
    except ImportError:
        skipped["lifelines"] = (
            "lifelines is not installed; skipping this engine "
            "(pip install lifelines to enable it)."
        )
    else:
        wf = WeibullFitter().fit(xa, ev)
        b, e = float(wf.rho_), float(wf.lambda_)
        estimates["lifelines"] = EngineEstimate(
            beta=b, eta=e, b10=quantile("weibull", {"beta": b, "eta": e}, 0.10)
        )

    try:
        from reliability.Fitters import (  # type: ignore[import-not-found]
            Fit_Weibull_2P,
        )
    except ImportError:
        skipped["reliability"] = (
            "the 'reliability' package is not installed; skipping this engine "
            "(pip install reliability to enable it)."
        )
    else:
        censored_times = xa[ev == 0]
        pkg_fit = Fit_Weibull_2P(
            failures=xa[ev == 1],
            right_censored=censored_times if len(censored_times) else None,
            show_probability_plot=False,
            print_results=False,
        )
        try:  # the package may open figures even with plotting disabled
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass
        b, e = float(pkg_fit.beta), float(pkg_fit.alpha)
        estimates["reliability"] = EngineEstimate(
            beta=b, eta=e, b10=quantile("weibull", {"beta": b, "eta": e}, 0.10)
        )

    max_rel = 0.0
    names = ("beta", "eta", "b10")
    engines = list(estimates)
    for i, eng_a in enumerate(engines):
        for eng_b in engines[i + 1 :]:
            for name in names:
                va = getattr(estimates[eng_a], name)
                vb = getattr(estimates[eng_b], name)
                rel = abs(va - vb) / max(abs(va), abs(vb))
                max_rel = max(max_rel, rel)
                if rel > tol:
                    raise CrossValidationError(
                        f"{name} disagrees between engines {eng_a!r} and "
                        f"{eng_b!r}: {va:.6g} vs {vb:.6g} "
                        f"({rel:.2%} > {tol:.0%}). Inspect the sample for "
                        "ties/heavy censoring, or loosen tol if a coarser "
                        "agreement is acceptable."
                    )
    return CrossValidationResult(
        estimates=estimates,
        skipped=skipped,
        tolerance=float(tol),
        max_relative_difference=float(max_rel),
    )
