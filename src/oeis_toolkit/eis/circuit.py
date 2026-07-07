"""Equivalent-circuit fitting of impedance spectra: R_s + (R ‖ CPE) arcs.

This module fits the series-resistance / R‖CPE ladder

    Z(omega) = R_s + sum_k  R_k / (1 + R_k * Q_k * (j*omega)**alpha_k),
    k = 1 .. n_arcs,

to a single impedance sweep by complex nonlinear least squares
(:func:`scipy.optimize.least_squares`, trust-region reflective with box
bounds). Real and imaginary residuals are stacked and, by default, weighted
by the impedance modulus ``|Z|`` at each frequency ("modulus weighting"),
so every frequency decade contributes comparably regardless of the absolute
impedance level.

The constant-phase element (CPE) has admittance ``Y = Q * (j*omega)**alpha``
with ``Q`` in ohm^-1 * s^alpha and ``0 < alpha <= 1`` (``alpha = 1`` recovers
an ideal capacitor). For each fitted arc the effective capacitance is
reported via the Hsu-Mansfeld relation

    omega_max = (R * Q)**(-1/alpha)   =>   C_eff = (R * Q)**(1/alpha) / R,

and arcs in a multi-arc fit are reported ordered by *descending*
characteristic frequency (arc 1 = highest-frequency process).

Model selection
---------------
:func:`fit_best_model` fits each candidate arc count to the *same* prepared
data set and keeps the model with the lowest corrected Akaike information
criterion,

    AICc = n * ln(RSS / n) + 2*k + 2*k*(k + 1) / (n - k - 1),

where ``n`` is the number of scalar residuals (2 x number of frequency
points, real and imaginary stacked) and ``k`` the number of free parameters.
AICc penalizes the extra three parameters of each additional arc, so a
second arc is retained only when it buys a genuine reduction in misfit.

Initial guesses
---------------
Starting values are derived from the data, not from tabulated constants:

* ``R_s``: the high-frequency intercept, estimated as ``min(Re Z)`` over the
  fitting window.
* Total arc span: ``max(Re Z) - min(Re Z)``.
* Single arc: the characteristic frequency is taken at the apex of the arc,
  i.e. the frequency of the ``-Im(Z)`` maximum.
* Two arcs: characteristic frequencies are placed at log-spaced quantiles of
  the measured frequency band, with the arc span partitioned heuristically
  between them. ``Q`` then follows from ``omega_max = (R*Q)**(-1/alpha)``.

Valley resistance
-----------------
:func:`valley_resistance` reports ``Re(Z)`` at the local minimum of
``-Im(Z)`` inside a caller-supplied frequency window. When the window
brackets the "valley" between two capacitive arcs, this is a model-free
proxy for the cumulative real resistance up to that arc boundary (ohmic plus
first-arc resistance) and is commonly used as an IR-correction resistance
without committing to an equivalent-circuit model. The window is
deliberately *not* defaulted: it depends on the instrument, cell geometry,
and chemistry, and must be chosen by inspecting representative spectra.

Conventions and scope
---------------------
* Inputs are the tabulated ``(frequency, Re(Z), -Im(Z))`` columns commonly
  exported by impedance instruments; internally ``Z = Re - j * (-Im)``.
* The module operates on one sweep at a time. If a data file concatenates
  several sweeps, split it upstream before fitting.
* Series inductance is not modeled. High-frequency inductive points
  (``-Im(Z) < 0``) can be excluded with ``drop_inductive_above_hz``.
* All failures raise :class:`CircuitFitError` (or :class:`ValueError` for
  malformed inputs) with actionable diagnostics -- functions never return
  silent NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

__all__ = [
    "ArcParameters",
    "CircuitFitError",
    "CircuitFitResult",
    "ValleyResistance",
    "circuit_impedance",
    "cpe_arc_impedance",
    "fit_best_model",
    "fit_circuit",
    "valley_resistance",
]

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

#: Default CPE exponent used to seed the optimizer (dimensionless, generic).
_ALPHA_SEED = 0.85

def _min_points(n_arcs: int) -> int:
    """Minimum number of frequency points required to fit an ``n_arcs``
    model: at least two scalar residuals per free parameter plus margin."""
    return 8 + 3 * n_arcs


class CircuitFitError(RuntimeError):
    """Raised when a circuit fit or spectrum reduction cannot succeed.

    The exception always carries a human-readable message describing what
    went wrong and what to check, plus a machine-readable ``diagnostics``
    mapping for unattended pipelines (e.g. point counts, solver status,
    candidate-model errors).

    Attributes
    ----------
    diagnostics : dict
        Context describing the failure. Keys depend on the failure site.
    """

    def __init__(
        self, message: str, *, diagnostics: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics: dict[str, object] = dict(diagnostics or {})


# --------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------
def cpe_arc_impedance(
    freq_hz: npt.ArrayLike, resistance_ohm: float, q: float, alpha: float
) -> ComplexArray:
    """Impedance of one R ‖ CPE arc.

    ``Z = R / (1 + R * Q * (j*omega)**alpha)`` with ``omega = 2*pi*f``.

    Parameters
    ----------
    freq_hz : array_like
        Frequencies in Hz (must be positive).
    resistance_ohm : float
        Arc resistance ``R`` in ohm.
    q : float
        CPE coefficient ``Q`` in ohm^-1 * s^alpha.
    alpha : float
        CPE exponent, ``0 < alpha <= 1``.

    Returns
    -------
    numpy.ndarray of complex
        Complex impedance at each frequency.
    """
    w = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    return np.asarray(
        resistance_ohm / (1.0 + resistance_ohm * q * (1j * w) ** alpha),
        dtype=complex,
    )


def circuit_impedance(
    freq_hz: npt.ArrayLike,
    r_series_ohm: float,
    arcs: Sequence[tuple[float, float, float]],
) -> ComplexArray:
    """Impedance of ``R_s`` in series with one or more R ‖ CPE arcs.

    Parameters
    ----------
    freq_hz : array_like
        Frequencies in Hz.
    r_series_ohm : float
        Series (ohmic) resistance in ohm.
    arcs : sequence of (resistance_ohm, q, alpha)
        Parameters of each R ‖ CPE arc.

    Returns
    -------
    numpy.ndarray of complex
        Complex impedance at each frequency.
    """
    f = np.asarray(freq_hz, dtype=float)
    z = np.full(f.shape, r_series_ohm, dtype=complex)
    for resistance_ohm, q, alpha in arcs:
        z = z + cpe_arc_impedance(f, resistance_ohm, q, alpha)
    return z


def _z_model(w: FloatArray, params: FloatArray, n_arcs: int) -> ComplexArray:
    """Evaluate the circuit at angular frequencies ``w`` from a flat
    parameter vector ``[R_s, R_1, Q_1, alpha_1, ...]``."""
    z = np.full(w.shape, params[0], dtype=complex)
    for k in range(n_arcs):
        r, q, a = params[1 + 3 * k : 4 + 3 * k]
        z = z + r / (1.0 + r * q * (1j * w) ** a)
    return z


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ArcParameters:
    """Fitted parameters of one R ‖ CPE arc.

    Attributes
    ----------
    resistance_ohm : float
        Arc resistance ``R`` in ohm.
    q : float
        CPE coefficient ``Q`` in ohm^-1 * s^alpha.
    alpha : float
        CPE exponent (dimensionless, ``0 < alpha <= 1``).
    resistance_std_ohm, q_std, alpha_std : float or None
        One-sigma uncertainties from the Gauss-Newton covariance estimate,
        or ``None`` when the covariance is unavailable or non-finite (e.g.
        rank-deficient Jacobian, parameter pinned at a bound).
    """

    resistance_ohm: float
    q: float
    alpha: float
    resistance_std_ohm: float | None = None
    q_std: float | None = None
    alpha_std: float | None = None

    @property
    def time_constant_s(self) -> float:
        """Characteristic relaxation time ``tau = (R*Q)**(1/alpha)`` in s."""
        return float((self.resistance_ohm * self.q) ** (1.0 / self.alpha))

    @property
    def characteristic_frequency_hz(self) -> float:
        """Frequency of the arc apex, ``f = 1 / (2*pi*tau)`` in Hz."""
        return float(1.0 / (2.0 * np.pi * self.time_constant_s))

    @property
    def effective_capacitance_f(self) -> float:
        """Effective capacitance ``C_eff = tau / R`` in farad
        (Hsu-Mansfeld / Brug form for an R ‖ CPE arc)."""
        return float(self.time_constant_s / self.resistance_ohm)


@dataclass(frozen=True)
class CircuitFitResult:
    """Result of a circuit fit (one model, or the AICc-selected winner).

    Attributes
    ----------
    n_arcs : int
        Number of R ‖ CPE arcs in the fitted model.
    r_series_ohm : float
        Fitted series resistance in ohm.
    r_series_std_ohm : float or None
        One-sigma uncertainty of ``r_series_ohm`` (see
        :class:`ArcParameters` for availability caveats).
    arcs : tuple of ArcParameters
        Fitted arcs, ordered by descending characteristic frequency
        (``arcs[0]`` is the highest-frequency process).
    rmse_pct : float
        Root-mean-square of the weighted residuals, in percent. With the
        default modulus weighting this is a relative misfit measure.
    aicc : float
        Corrected Akaike information criterion of this model.
    n_points_used : int
        Number of frequency points that entered the fit after windowing
        and inductive-point removal.
    aicc_by_model : dict[int, float]
        AICc of every candidate model that converged (keyed by arc count).
        Contains only this model's entry when produced by
        :func:`fit_circuit`.
    model_errors : dict[int, str]
        Failure messages for candidate models that did not converge
        (populated by :func:`fit_best_model`).
    """

    n_arcs: int
    r_series_ohm: float
    r_series_std_ohm: float | None
    arcs: tuple[ArcParameters, ...]
    rmse_pct: float
    aicc: float
    n_points_used: int
    aicc_by_model: dict[int, float] = field(default_factory=dict)
    model_errors: dict[int, str] = field(default_factory=dict)

    @property
    def r_total_ohm(self) -> float:
        """Series resistance plus the sum of all arc resistances, in ohm."""
        return float(
            self.r_series_ohm + sum(a.resistance_ohm for a in self.arcs)
        )

    def impedance(self, freq_hz: npt.ArrayLike) -> ComplexArray:
        """Evaluate the fitted model at the given frequencies (Hz)."""
        return circuit_impedance(
            freq_hz,
            self.r_series_ohm,
            [(a.resistance_ohm, a.q, a.alpha) for a in self.arcs],
        )


@dataclass(frozen=True)
class ValleyResistance:
    """Model-free valley-resistance reduction of one spectrum.

    ``resistance_ohm`` is ``Re(Z)`` at the point of minimum ``-Im(Z)``
    inside the caller-supplied frequency window. When the window brackets
    the valley between two capacitive arcs, this approximates the
    cumulative real resistance up to that arc boundary and serves as a
    generic IR-correction proxy.

    Attributes
    ----------
    resistance_ohm : float
        ``Re(Z)`` at the valley point, in ohm.
    frequency_hz : float
        Frequency of the valley point, in Hz.
    neg_imag_ohm : float
        ``-Im(Z)`` at the valley point, in ohm (the minimized quantity).
    n_points_in_window : int
        Number of finite data points found inside the window.
    """

    resistance_ohm: float
    frequency_hz: float
    neg_imag_ohm: float
    n_points_in_window: int


# --------------------------------------------------------------------------
# Input preparation
# --------------------------------------------------------------------------
def _as_1d_float(name: str, values: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"'{name}' must be one-dimensional, got shape {arr.shape}."
        )
    return arr


def _prepare_spectrum(
    freq_hz: npt.ArrayLike,
    z_re_ohm: npt.ArrayLike,
    z_neg_im_ohm: npt.ArrayLike,
    *,
    f_min_hz: float | None,
    f_max_hz: float | None,
    drop_inductive_above_hz: float | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate, window, and clean one sweep. Returns (f, re, neg_im)."""
    f = _as_1d_float("freq_hz", freq_hz)
    re_ = _as_1d_float("z_re_ohm", z_re_ohm)
    nim = _as_1d_float("z_neg_im_ohm", z_neg_im_ohm)
    if not (f.size == re_.size == nim.size):
        raise ValueError(
            "freq_hz, z_re_ohm and z_neg_im_ohm must have equal lengths, "
            f"got {f.size}, {re_.size}, {nim.size}."
        )
    if f_min_hz is not None and f_max_hz is not None and f_min_hz >= f_max_hz:
        raise ValueError(
            f"f_min_hz ({f_min_hz}) must be smaller than f_max_hz "
            f"({f_max_hz})."
        )
    mask = np.isfinite(f) & np.isfinite(re_) & np.isfinite(nim) & (f > 0)
    if f_max_hz is not None:
        mask &= f <= f_max_hz
    if f_min_hz is not None:
        mask &= f >= f_min_hz
    if drop_inductive_above_hz is not None:
        # Series inductance is not modeled: exclude high-frequency points
        # that have crossed into the inductive half-plane.
        mask &= ~((nim < 0) & (f > drop_inductive_above_hz))
    return f[mask], re_[mask], nim[mask]


# --------------------------------------------------------------------------
# Fitting internals
# --------------------------------------------------------------------------
@dataclass
class _FitOutput:
    """Internal container passed between the solver and result builders."""

    n_arcs: int
    params: FloatArray
    stds: FloatArray | None
    aicc: float
    rmse_pct: float
    n_used: int


def _initial_guess(
    f: FloatArray,
    re_: FloatArray,
    nim: FloatArray,
    n_arcs: int,
    alpha_bounds: tuple[float, float],
) -> tuple[list[float], list[float], list[float]]:
    """Data-driven starting point and box bounds.

    Returns ``(p0, lower, upper)`` for the flat parameter vector
    ``[R_s, R_1, Q_1, alpha_1, ...]``.
    """
    alpha0 = float(np.clip(_ALPHA_SEED, alpha_bounds[0], alpha_bounds[1]))
    rs0 = float(np.min(re_))  # high-frequency intercept estimate
    r_span = float(np.max(re_)) - rs0
    if not np.isfinite(r_span) or r_span <= 0:
        raise CircuitFitError(
            "Spectrum has no resolvable arc span (max(Re Z) <= min(Re Z)). "
            "Check the column sign convention and the fitting window.",
            diagnostics={
                "r_hf_estimate_ohm": rs0,
                "re_span_ohm": r_span,
                "n_points": int(f.size),
            },
        )

    p0: list[float] = [max(rs0, 1e-6 * r_span)]
    lower: list[float] = [0.0]
    upper: list[float] = [np.inf]

    f_hi = float(np.max(f))
    f_lo = float(np.min(f))

    if n_arcs == 1:
        # Arc apex heuristic: characteristic frequency at the -Im(Z) maximum.
        i_apex = int(np.argmax(nim))
        if nim[i_apex] > 0:
            char_freqs = [float(f[i_apex])]
        else:
            char_freqs = [float(np.sqrt(f_hi * f_lo))]
        shares = [1.0]
    else:
        # Log-spaced quantiles of the measured band, high frequency first.
        char_freqs = [
            float(
                np.exp(
                    np.interp(
                        k + 0.5, [0.0, n_arcs], np.log([f_hi, f_lo])
                    )
                )
            )
            for k in range(n_arcs)
        ]
        if n_arcs == 2:
            shares = [0.35, 0.65]
        else:
            shares = [1.0 / n_arcs] * n_arcs

    for fk, share in zip(char_freqs, shares):
        r_k = max(r_span * share, 1e-3 * r_span)
        q_k = 1.0 / (r_k * (2.0 * np.pi * fk) ** alpha0)
        p0 += [r_k, q_k, alpha0]
        lower += [1e-6 * r_span, 1e-16, alpha_bounds[0]]
        upper += [1e6 * r_span, 1e6, alpha_bounds[1]]

    p0 = [float(np.clip(p, lo, hi)) for p, lo, hi in zip(p0, lower, upper)]
    return p0, lower, upper


def _parameter_std(jac: FloatArray, resid: FloatArray) -> FloatArray | None:
    """One-sigma parameter uncertainties from the Gauss-Newton covariance
    ``cov = pinv(J^T J) * RSS / (n - k)``. Returns None when unavailable."""
    n, k = resid.size, jac.shape[1]
    dof = n - k
    if dof <= 0:
        return None
    rss = float(np.dot(resid, resid))
    try:
        cov = np.linalg.pinv(jac.T @ jac) * (rss / dof)
    except np.linalg.LinAlgError:
        return None
    diag = np.diag(cov).copy()
    diag[diag < 0] = np.nan
    return np.sqrt(diag)


def _std_or_none(value: float | np.floating) -> float | None:
    v = float(value)
    return v if np.isfinite(v) else None


def _fit_prepared(
    f: FloatArray,
    re_: FloatArray,
    nim: FloatArray,
    n_arcs: int,
    *,
    alpha_bounds: tuple[float, float],
    weighting: Literal["modulus", "unit"],
    max_nfev: int,
) -> _FitOutput:
    """Fit one model to already-prepared data. Raises CircuitFitError."""
    needed = _min_points(n_arcs)
    if f.size < needed:
        raise CircuitFitError(
            f"Too few usable points for a {n_arcs}-arc fit: {f.size} "
            f"remain after windowing but at least {needed} are required. "
            "Widen the frequency window or use fewer arcs.",
            diagnostics={
                "n_points": int(f.size),
                "n_required": needed,
                "n_arcs": n_arcs,
            },
        )

    w = 2.0 * np.pi * f
    z_exp = re_ - 1j * nim  # -Im column convention: Im(Z) = -z_neg_im_ohm

    if weighting == "modulus":
        wgt = np.abs(z_exp)
        floor = float(np.max(wgt))
        if floor <= 0:
            raise CircuitFitError(
                "All impedance magnitudes are zero; nothing to fit.",
                diagnostics={"n_points": int(f.size)},
            )
        wgt = np.where(wgt > 0, wgt, floor)
    elif weighting == "unit":
        wgt = np.ones_like(f)
    else:  # pragma: no cover - guarded by the public API
        raise ValueError(
            f"Unknown weighting '{weighting}'; use 'modulus' or 'unit'."
        )

    p0, lower, upper = _initial_guess(f, re_, nim, n_arcs, alpha_bounds)

    def resid(p: FloatArray) -> FloatArray:
        z = _z_model(w, p, n_arcs)
        return np.concatenate(
            [(z.real - z_exp.real) / wgt, (z.imag - z_exp.imag) / wgt]
        )

    try:
        sol = least_squares(
            resid, p0, bounds=(lower, upper), max_nfev=max_nfev
        )
    except Exception as exc:
        raise CircuitFitError(
            f"Least-squares solver raised for the {n_arcs}-arc model: "
            f"{exc}. Check the spectrum for gross artifacts (sign flips, "
            "duplicated frequencies) before retrying.",
            diagnostics={"n_arcs": n_arcs, "n_points": int(f.size)},
        ) from exc

    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise CircuitFitError(
            f"The {n_arcs}-arc fit did not converge "
            f"(solver status {sol.status}: {sol.message}). "
            "Consider adjusting the frequency window or alpha bounds.",
            diagnostics={
                "n_arcs": n_arcs,
                "status": int(sol.status),
                "message": str(sol.message),
                "cost": float(sol.cost),
                "nfev": int(sol.nfev),
                "n_points": int(f.size),
            },
        )

    r = resid(sol.x)
    n = r.size
    k = sol.x.size
    rss = float(np.dot(r, r))
    # AICc; the min-points guard above guarantees n - k - 1 > 0.
    aicc = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
    return _FitOutput(
        n_arcs=n_arcs,
        params=np.asarray(sol.x, dtype=float),
        stds=_parameter_std(sol.jac, r),
        aicc=float(aicc),
        rmse_pct=float(100.0 * np.sqrt(rss / n)),
        n_used=int(f.size),
    )


def _to_result(
    out: _FitOutput,
    aicc_by_model: dict[int, float],
    model_errors: dict[int, str],
) -> CircuitFitResult:
    """Assemble the public result, ordering arcs by characteristic
    frequency (highest first)."""
    stds = out.stds
    arcs: list[ArcParameters] = []
    for k in range(out.n_arcs):
        j = 1 + 3 * k
        arcs.append(
            ArcParameters(
                resistance_ohm=float(out.params[j]),
                q=float(out.params[j + 1]),
                alpha=float(out.params[j + 2]),
                resistance_std_ohm=(
                    _std_or_none(stds[j]) if stds is not None else None
                ),
                q_std=(
                    _std_or_none(stds[j + 1]) if stds is not None else None
                ),
                alpha_std=(
                    _std_or_none(stds[j + 2]) if stds is not None else None
                ),
            )
        )
    arcs.sort(key=lambda a: a.characteristic_frequency_hz, reverse=True)
    return CircuitFitResult(
        n_arcs=out.n_arcs,
        r_series_ohm=float(out.params[0]),
        r_series_std_ohm=(
            _std_or_none(stds[0]) if stds is not None else None
        ),
        arcs=tuple(arcs),
        rmse_pct=out.rmse_pct,
        aicc=out.aicc,
        n_points_used=out.n_used,
        aicc_by_model=dict(aicc_by_model),
        model_errors=dict(model_errors),
    )


def _validate_common(
    alpha_bounds: tuple[float, float],
    weighting: str,
    max_nfev: int,
) -> None:
    lo, hi = alpha_bounds
    if not (0.0 < lo < hi <= 1.0):
        raise ValueError(
            f"alpha_bounds must satisfy 0 < lower < upper <= 1, got "
            f"{alpha_bounds}."
        )
    if weighting not in ("modulus", "unit"):
        raise ValueError(
            f"weighting must be 'modulus' or 'unit', got '{weighting}'."
        )
    if max_nfev < 1:
        raise ValueError(f"max_nfev must be positive, got {max_nfev}.")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def fit_circuit(
    freq_hz: npt.ArrayLike,
    z_re_ohm: npt.ArrayLike,
    z_neg_im_ohm: npt.ArrayLike,
    *,
    n_arcs: int = 2,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    drop_inductive_above_hz: float | None = None,
    alpha_bounds: tuple[float, float] = (0.3, 1.0),
    weighting: Literal["modulus", "unit"] = "modulus",
    max_nfev: int = 4000,
) -> CircuitFitResult:
    """Fit ``R_s + n_arcs x (R ‖ CPE)`` to one impedance sweep.

    Parameters
    ----------
    freq_hz : array_like
        Frequencies in Hz (one sweep; order is irrelevant).
    z_re_ohm : array_like
        Real part of the impedance, ohm.
    z_neg_im_ohm : array_like
        Negative imaginary part of the impedance (``-Im(Z)``), ohm --
        the sign convention of typical instrument exports.
    n_arcs : int, optional
        Number of R ‖ CPE arcs to fit (default 2).
    f_min_hz, f_max_hz : float, optional
        Fitting window in Hz. ``None`` (default) uses all finite points.
        Supply these to exclude instrument artifacts at the band edges;
        appropriate values depend on the setup and must come from the
        caller.
    drop_inductive_above_hz : float, optional
        If given, points with ``-Im(Z) < 0`` at frequencies above this
        threshold are excluded (series inductance is not modeled).
        ``None`` (default) keeps all points.
    alpha_bounds : tuple of float, optional
        Box bounds for every CPE exponent (default ``(0.3, 1.0)``).
    weighting : {'modulus', 'unit'}, optional
        Residual weighting. ``'modulus'`` (default) divides each residual
        by ``|Z|``; ``'unit'`` uses unweighted residuals.
    max_nfev : int, optional
        Maximum number of residual evaluations for the solver.

    Returns
    -------
    CircuitFitResult
        Fitted parameters, uncertainties, misfit, and AICc.

    Raises
    ------
    CircuitFitError
        If too few points remain after windowing, the arc span is not
        resolvable, or the solver fails to converge. The exception's
        ``diagnostics`` attribute describes the failure.
    ValueError
        If inputs are malformed (shape mismatch, invalid window or
        options).
    """
    if n_arcs < 1:
        raise ValueError(f"n_arcs must be >= 1, got {n_arcs}.")
    _validate_common(alpha_bounds, weighting, max_nfev)
    f, re_, nim = _prepare_spectrum(
        freq_hz,
        z_re_ohm,
        z_neg_im_ohm,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        drop_inductive_above_hz=drop_inductive_above_hz,
    )
    out = _fit_prepared(
        f,
        re_,
        nim,
        n_arcs,
        alpha_bounds=alpha_bounds,
        weighting=weighting,
        max_nfev=max_nfev,
    )
    return _to_result(out, {out.n_arcs: out.aicc}, {})


def fit_best_model(
    freq_hz: npt.ArrayLike,
    z_re_ohm: npt.ArrayLike,
    z_neg_im_ohm: npt.ArrayLike,
    *,
    candidate_n_arcs: Sequence[int] = (1, 2),
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    drop_inductive_above_hz: float | None = None,
    alpha_bounds: tuple[float, float] = (0.3, 1.0),
    weighting: Literal["modulus", "unit"] = "modulus",
    max_nfev: int = 4000,
) -> CircuitFitResult:
    """Fit each candidate arc count and keep the lowest-AICc model.

    All candidates are fitted to the *same* prepared data (identical
    windowing and cleaning), which is required for the AICc values to be
    comparable. Candidates that fail to converge are recorded in the
    result's ``model_errors`` and skipped; the fit only raises if *every*
    candidate fails.

    Parameters
    ----------
    freq_hz, z_re_ohm, z_neg_im_ohm : array_like
        See :func:`fit_circuit`.
    candidate_n_arcs : sequence of int, optional
        Arc counts to compare (default ``(1, 2)``).
    f_min_hz, f_max_hz, drop_inductive_above_hz, alpha_bounds, weighting, \
max_nfev
        See :func:`fit_circuit`.

    Returns
    -------
    CircuitFitResult
        The winning model. ``aicc_by_model`` holds the AICc of every
        candidate that converged; ``model_errors`` holds failure messages
        for those that did not.

    Raises
    ------
    CircuitFitError
        If every candidate model fails. ``diagnostics['model_errors']``
        maps each arc count to its failure message.
    ValueError
        If inputs or options are malformed.
    """
    candidates = list(candidate_n_arcs)
    if not candidates or any(c < 1 for c in candidates):
        raise ValueError(
            f"candidate_n_arcs must be a non-empty sequence of ints >= 1, "
            f"got {candidate_n_arcs!r}."
        )
    _validate_common(alpha_bounds, weighting, max_nfev)
    f, re_, nim = _prepare_spectrum(
        freq_hz,
        z_re_ohm,
        z_neg_im_ohm,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        drop_inductive_above_hz=drop_inductive_above_hz,
    )

    outputs: dict[int, _FitOutput] = {}
    aiccs: dict[int, float] = {}
    errors: dict[int, str] = {}
    for n_arcs in candidates:
        try:
            out = _fit_prepared(
                f,
                re_,
                nim,
                n_arcs,
                alpha_bounds=alpha_bounds,
                weighting=weighting,
                max_nfev=max_nfev,
            )
        except CircuitFitError as exc:
            errors[n_arcs] = str(exc)
            continue
        outputs[n_arcs] = out
        aiccs[n_arcs] = out.aicc

    if not outputs:
        raise CircuitFitError(
            "Every candidate model failed to fit this spectrum. "
            "Inspect the per-model messages in diagnostics['model_errors'] "
            "and verify the data window and sign convention.",
            diagnostics={
                "model_errors": errors,
                "n_points": int(f.size),
                "candidates": candidates,
            },
        )

    best = min(outputs.values(), key=lambda o: o.aicc)
    return _to_result(best, aiccs, errors)


def valley_resistance(
    freq_hz: npt.ArrayLike,
    z_re_ohm: npt.ArrayLike,
    z_neg_im_ohm: npt.ArrayLike,
    *,
    f_min_hz: float,
    f_max_hz: float,
) -> ValleyResistance:
    """``Re(Z)`` at the ``-Im(Z)`` minimum within a frequency window.

    This is a model-free "valley resistance": when the supplied window
    brackets the dip between two capacitive arcs on the Nyquist plot, the
    real impedance at that dip approximates the cumulative resistance of
    all processes faster than the window (ohmic plus first-arc), making it
    a generic IR-correction proxy that does not depend on an
    equivalent-circuit fit.

    The window is intentionally required rather than defaulted: the valley
    position depends on the instrument, cell geometry, and chemistry, and
    should be chosen by inspecting representative spectra.

    Parameters
    ----------
    freq_hz, z_re_ohm, z_neg_im_ohm : array_like
        One sweep, same conventions as :func:`fit_circuit`.
    f_min_hz, f_max_hz : float
        Frequency window (Hz) that brackets the inter-arc valley.
        Required keyword arguments; ``f_min_hz < f_max_hz`` and both
        positive.

    Returns
    -------
    ValleyResistance
        Valley-point resistance, frequency, ``-Im(Z)`` value, and the
        number of points searched.

    Raises
    ------
    CircuitFitError
        If no finite data point falls inside the window.
    ValueError
        If inputs are malformed or the window is invalid.
    """
    if not (f_min_hz > 0 and f_max_hz > 0 and f_min_hz < f_max_hz):
        raise ValueError(
            "Valley window must satisfy 0 < f_min_hz < f_max_hz, got "
            f"f_min_hz={f_min_hz}, f_max_hz={f_max_hz}."
        )
    f = _as_1d_float("freq_hz", freq_hz)
    re_ = _as_1d_float("z_re_ohm", z_re_ohm)
    nim = _as_1d_float("z_neg_im_ohm", z_neg_im_ohm)
    if not (f.size == re_.size == nim.size):
        raise ValueError(
            "freq_hz, z_re_ohm and z_neg_im_ohm must have equal lengths, "
            f"got {f.size}, {re_.size}, {nim.size}."
        )
    mask = (
        np.isfinite(f)
        & np.isfinite(re_)
        & np.isfinite(nim)
        & (f >= f_min_hz)
        & (f <= f_max_hz)
    )
    n_in = int(np.count_nonzero(mask))
    if n_in == 0:
        raise CircuitFitError(
            f"No finite data points inside the valley window "
            f"[{f_min_hz:g}, {f_max_hz:g}] Hz. Supply a window that "
            "overlaps the measured band and brackets the inter-arc "
            "-Im(Z) minimum.",
            diagnostics={
                "f_min_hz": float(f_min_hz),
                "f_max_hz": float(f_max_hz),
                "data_f_min_hz": float(np.nanmin(f)) if f.size else None,
                "data_f_max_hz": float(np.nanmax(f)) if f.size else None,
                "n_points_total": int(f.size),
            },
        )
    idx_window = np.flatnonzero(mask)
    i = idx_window[int(np.argmin(nim[idx_window]))]
    return ValleyResistance(
        resistance_ohm=float(re_[i]),
        frequency_hz=float(f[i]),
        neg_imag_ohm=float(nim[i]),
        n_points_in_window=n_in,
    )
