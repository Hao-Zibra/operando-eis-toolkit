"""Tikhonov-regularized Distribution of Relaxation Times (DRT).

Model
-----
An impedance spectrum dominated by parallel relaxation processes is
represented as

    Z(omega) = R_inf + integral  gamma(ln tau) / (1 + j*omega*tau)  d ln tau,

where ``R_inf`` is the high-frequency (instantaneous) resistance and
``gamma(ln tau) >= 0`` distributes the polarization resistance over
relaxation times: ``R_pol = integral gamma d ln tau``.

Discretization
--------------
The distribution is discretized with piecewise-constant elements on a
log-spaced relaxation-time grid (:func:`tau_grid`): ``points_per_decade``
nodes per decade covering ``[1/(2*pi*f_max), 1/(2*pi*f_min)]`` extended by
``extension_decades`` on both sides so that peaks near the band edges are
not truncated. The uniform ``d ln tau`` element width is absorbed into the
unknowns, i.e. the solution vector is ``x_k = gamma_k * d(ln tau)``, so
``sum(x)`` directly equals the polarization resistance and the reported
``gamma`` integrates by simple summation.

Regularization and solver
-------------------------
The discretized problem is solved as non-negative least squares (NNLS) on
the stacked real/imaginary system augmented with a curvature penalty:

    min_x  || A x - b ||^2  +  lambda^2 || L x ||^2     s.t.  x >= 0,

implemented by appending ``lambda * s * L`` rows to the design matrix,
where ``L`` is the second-difference operator over the tau grid and
``s = median(|Z|)`` scales the penalty so that ``lambda`` is dimensionless
with respect to the impedance level. ``R_inf`` is fitted jointly as an
additional non-negative, *unpenalized* column. The non-negativity
constraint plays the role of the positivity prior used in RBF-based DRT
formulations; the second-difference penalty suppresses spurious
oscillations of the piecewise-constant solution.

Regularization-parameter selection
----------------------------------
When ``lam`` is not fixed by the caller, it is selected per spectrum by an
L-curve criterion: the problem is solved on a log-spaced ``lambda`` grid,
the path of (log residual norm, log solution norm) points is traced, and
the grid point of maximum discrete curvature (largest normalized
cross-product turn between consecutive path segments) is taken as the
corner. The residual norm is the relative RMS misfit; the solution norm is
the squared 2-norm of ``gamma``.

Interpretation caveats
----------------------
Peak *positions* (relaxation times) and integrated peak *areas*
(resistances) are the robust outputs of a regularized DRT; detailed peak
shapes are not, as they depend on the regularization strength. Series
inductance is not modeled -- exclude high-frequency inductive points via
``drop_inductive_above_hz`` (or an ``f_max_hz`` window) before inversion.

All failures raise :class:`DRTError` (or :class:`ValueError` for malformed
inputs) with actionable messages -- functions never return silent NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls

__all__ = [
    "DRTError",
    "DRTPeak",
    "DRTResult",
    "compute_drt",
    "find_drt_peaks",
    "region_resistance",
    "tau_grid",
]

FloatArray = npt.NDArray[np.float64]

#: Default lambda grid searched by the L-curve criterion (dimensionless,
#: thanks to the median-|Z| scaling of the penalty rows).
_DEFAULT_LAMBDA_GRID = (-4.0, 0.0, 9)  # log10 start, log10 stop, count


class DRTError(RuntimeError):
    """Raised when a DRT inversion cannot produce a trustworthy result.

    Carries a human-readable message plus a machine-readable
    ``diagnostics`` mapping for unattended pipelines.

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
# Result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DRTResult:
    """Result of one DRT inversion.

    Attributes
    ----------
    tau_s : numpy.ndarray
        Log-spaced relaxation-time grid, seconds.
    gamma_ohm : numpy.ndarray
        Distributed resistance per grid element (``gamma_k * d ln tau``),
        ohm. ``gamma_ohm.sum()`` equals ``r_pol_ohm``.
    r_inf_ohm : float
        Fitted high-frequency resistance, ohm.
    r_pol_ohm : float
        Total polarization resistance ``sum(gamma_ohm)``, ohm.
    lam : float
        Regularization parameter used (fixed by the caller or selected by
        the L-curve criterion).
    residual_pct : float
        Relative RMS misfit of the reconstructed impedance, percent of
        the median impedance modulus.
    n_points_used : int
        Number of frequency points that entered the inversion.
    lambda_grid : numpy.ndarray or None
        The lambda values searched by the L-curve criterion, or ``None``
        when ``lam`` was fixed by the caller.
    lcurve_log_residual : numpy.ndarray or None
        Log residual-norm coordinate of each L-curve point (same order as
        ``lambda_grid``), or ``None`` when ``lam`` was fixed.
    lcurve_log_solution_norm : numpy.ndarray or None
        Log solution-norm coordinate of each L-curve point, or ``None``
        when ``lam`` was fixed.
    """

    tau_s: FloatArray
    gamma_ohm: FloatArray
    r_inf_ohm: float
    r_pol_ohm: float
    lam: float
    residual_pct: float
    n_points_used: int
    lambda_grid: FloatArray | None = None
    lcurve_log_residual: FloatArray | None = None
    lcurve_log_solution_norm: FloatArray | None = None


@dataclass(frozen=True)
class DRTPeak:
    """One peak of the distribution.

    Attributes
    ----------
    tau_s : float
        Relaxation time at the local maximum of ``gamma``, seconds.
    resistance_ohm : float
        Resistance integrated over the peak's watershed region (from the
        minimum on its left to the minimum on its right), ohm.
    fraction : float
        ``resistance_ohm`` as a fraction of the total polarization
        resistance.
    """

    tau_s: float
    resistance_ohm: float
    fraction: float


# --------------------------------------------------------------------------
# Grid and design matrix
# --------------------------------------------------------------------------
def tau_grid(
    f_min_hz: float,
    f_max_hz: float,
    *,
    points_per_decade: int = 7,
    extension_decades: float = 1.0,
) -> FloatArray:
    """Log-spaced relaxation-time grid for a measured frequency band.

    The grid covers ``[1/(2*pi*f_max_hz), 1/(2*pi*f_min_hz)]`` extended by
    ``extension_decades`` decades on both sides, with
    ``points_per_decade`` nodes per decade.

    Parameters
    ----------
    f_min_hz, f_max_hz : float
        Lowest and highest measured frequencies, Hz (both positive,
        ``f_min_hz < f_max_hz``).
    points_per_decade : int, optional
        Grid resolution (default 7). Higher values resolve closely spaced
        processes at the cost of a harder-to-regularize problem.
    extension_decades : float, optional
        Symmetric extension beyond the measured band, in decades
        (default 1.0), so edge peaks are not truncated.

    Returns
    -------
    numpy.ndarray
        Relaxation times in seconds, ascending.

    Raises
    ------
    ValueError
        If the frequency band or grid options are invalid.
    """
    if not (f_min_hz > 0 and f_max_hz > 0 and f_min_hz < f_max_hz):
        raise ValueError(
            "Frequency band must satisfy 0 < f_min_hz < f_max_hz, got "
            f"f_min_hz={f_min_hz}, f_max_hz={f_max_hz}."
        )
    if points_per_decade < 1:
        raise ValueError(
            f"points_per_decade must be >= 1, got {points_per_decade}."
        )
    if extension_decades < 0:
        raise ValueError(
            f"extension_decades must be >= 0, got {extension_decades}."
        )
    ext = 10.0 ** extension_decades
    t_min = 1.0 / (2.0 * np.pi * f_max_hz) / ext
    t_max = 1.0 / (2.0 * np.pi * f_min_hz) * ext
    n = int(np.ceil(np.log10(t_max / t_min) * points_per_decade)) + 1
    return np.logspace(np.log10(t_min), np.log10(t_max), n)


def _design(w: FloatArray, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Real and imaginary kernel matrices for ``x_k = gamma_k * d ln tau``
    (piecewise-constant elements): entries of ``1 / (1 + j*w*tau)``."""
    wt = np.outer(w, tau)
    denom = 1.0 + wt**2
    return 1.0 / denom, -wt / denom


def _second_diff(n: int) -> FloatArray:
    """Second-difference (discrete curvature) operator, shape (n-2, n)."""
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
    return L


def _solve(
    w: FloatArray, z: npt.NDArray[np.complex128], tau: FloatArray, lam: float
) -> tuple[float, FloatArray, float]:
    """Solve the augmented NNLS system for one lambda.

    Returns ``(r_inf, gamma, residual_pct)`` where ``residual_pct`` is the
    relative RMS misfit in percent of the median impedance modulus.
    """
    a_re, a_im = _design(w, tau)
    n_tau = len(tau)
    ones = np.ones((len(w), 1))  # R_inf column (real part only)
    a_top = np.hstack([ones, a_re])
    a_bot = np.hstack([np.zeros((len(w), 1)), a_im])
    penalty = np.hstack([np.zeros((n_tau - 2, 1)), _second_diff(n_tau)])
    scale = float(np.median(np.abs(z)))
    if scale <= 0 or not np.isfinite(scale):
        raise DRTError(
            "Median impedance modulus is zero or non-finite; the spectrum "
            "cannot be scaled for regularization. Check the input columns.",
            diagnostics={"median_abs_z": scale, "n_points": int(len(w))},
        )
    a = np.vstack([a_top, a_bot, lam * scale * penalty])
    b = np.concatenate([z.real, z.imag, np.zeros(n_tau - 2)])
    try:
        x, _ = nnls(a, b, maxiter=10 * a.shape[1])
    except Exception as exc:
        raise DRTError(
            f"NNLS failed to converge (lambda={lam:g}): {exc}. "
            "Try a coarser tau grid (lower points_per_decade) or a larger "
            "regularization parameter.",
            diagnostics={
                "lam": float(lam),
                "n_tau": int(n_tau),
                "n_points": int(len(w)),
            },
        ) from exc
    r_inf, gamma = float(x[0]), x[1:]
    fit = (a_top @ x) + 1j * (a_bot @ x)
    residual_pct = float(
        np.sqrt(np.mean(np.abs(fit - z) ** 2)) / scale * 100.0
    )
    return r_inf, gamma, residual_pct


def _lcurve_corner(points: FloatArray) -> int:
    """Index of maximum discrete curvature along the (log residual,
    log solution norm) path -- the normalized cross product of consecutive
    path segments, maximized over interior points."""
    best_k, best_c = len(points) // 2, -np.inf
    for k in range(1, len(points) - 1):
        v1 = points[k] - points[k - 1]
        v2 = points[k + 1] - points[k]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1e-12
        c = cross / denom
        if c > best_c:
            best_c, best_k = c, k
    return best_k


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def compute_drt(
    freq_hz: npt.ArrayLike,
    z_re_ohm: npt.ArrayLike,
    z_neg_im_ohm: npt.ArrayLike,
    *,
    lam: float | None = None,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    drop_inductive_above_hz: float | None = None,
    points_per_decade: int = 7,
    extension_decades: float = 1.0,
    lambda_grid: npt.ArrayLike | None = None,
    min_points: int = 12,
) -> DRTResult:
    """Compute the DRT of one impedance sweep.

    Parameters
    ----------
    freq_hz : array_like
        Frequencies in Hz (one sweep; order is irrelevant).
    z_re_ohm : array_like
        Real part of the impedance, ohm.
    z_neg_im_ohm : array_like
        Negative imaginary part of the impedance (``-Im(Z)``), ohm --
        the sign convention of typical instrument exports.
    lam : float, optional
        Fixed regularization parameter. ``None`` (default) selects it per
        spectrum by the L-curve criterion (see module docstring).
    f_min_hz, f_max_hz : float, optional
        Analysis window in Hz. ``None`` (default) uses all finite points.
        Supply these to exclude band-edge artifacts; appropriate values
        depend on the setup and must come from the caller.
    drop_inductive_above_hz : float, optional
        If given, points with ``-Im(Z) < 0`` at frequencies above this
        threshold are excluded (series inductance is not modeled).
        ``None`` (default) keeps all points.
    points_per_decade : int, optional
        Tau-grid resolution (default 7); see :func:`tau_grid`.
    extension_decades : float, optional
        Tau-grid extension beyond the measured band (default 1.0); see
        :func:`tau_grid`.
    lambda_grid : array_like, optional
        Lambda values searched by the L-curve criterion. Defaults to nine
        log-spaced values spanning 1e-4 to 1. Ignored when ``lam`` is
        fixed.
    min_points : int, optional
        Minimum number of usable frequency points (default 12).

    Returns
    -------
    DRTResult
        Grid, distribution, ``R_inf``, ``R_pol``, the lambda used, misfit,
        and (when lambda was auto-selected) the L-curve trace.

    Raises
    ------
    DRTError
        If too few points remain after windowing, the spectrum cannot be
        scaled, or the NNLS solver fails.
    ValueError
        If inputs or options are malformed.
    """
    f = np.asarray(freq_hz, dtype=float)
    re_ = np.asarray(z_re_ohm, dtype=float)
    nim = np.asarray(z_neg_im_ohm, dtype=float)
    if f.ndim != 1 or not (f.size == re_.size == nim.size):
        raise ValueError(
            "freq_hz, z_re_ohm and z_neg_im_ohm must be one-dimensional "
            f"and of equal length, got shapes {f.shape}, {re_.shape}, "
            f"{nim.shape}."
        )
    if f_min_hz is not None and f_max_hz is not None and f_min_hz >= f_max_hz:
        raise ValueError(
            f"f_min_hz ({f_min_hz}) must be smaller than f_max_hz "
            f"({f_max_hz})."
        )
    if lam is not None and not (np.isfinite(lam) and lam > 0):
        raise ValueError(f"lam must be finite and positive, got {lam}.")
    if min_points < 4:
        raise ValueError(f"min_points must be >= 4, got {min_points}.")

    z = re_ - 1j * nim  # -Im column convention: Im(Z) = -z_neg_im_ohm
    mask = (
        np.isfinite(f)
        & np.isfinite(z.real)
        & np.isfinite(z.imag)
        & (f > 0)
    )
    if f_max_hz is not None:
        mask &= f <= f_max_hz
    if f_min_hz is not None:
        mask &= f >= f_min_hz
    if drop_inductive_above_hz is not None:
        mask &= ~((nim < 0) & (f > drop_inductive_above_hz))
    f, z = f[mask], z[mask]
    if f.size < min_points:
        raise DRTError(
            f"Too few usable points for a DRT inversion: {f.size} remain "
            f"after windowing but at least {min_points} are required. "
            "Widen the frequency window or lower min_points.",
            diagnostics={
                "n_points": int(f.size),
                "n_required": int(min_points),
            },
        )

    order = np.argsort(f)[::-1]  # high to low frequency
    f, z = f[order], z[order]
    w = 2.0 * np.pi * f
    tau = tau_grid(
        float(f.min()),
        float(f.max()),
        points_per_decade=points_per_decade,
        extension_decades=extension_decades,
    )

    lams_out: FloatArray | None = None
    rho_out: FloatArray | None = None
    eta_out: FloatArray | None = None
    if lam is None:
        if lambda_grid is None:
            lo, hi, count = _DEFAULT_LAMBDA_GRID
            lams = np.logspace(lo, hi, count)
        else:
            lams = np.sort(np.asarray(lambda_grid, dtype=float))
            if lams.ndim != 1 or lams.size < 3 or np.any(lams <= 0):
                raise ValueError(
                    "lambda_grid must contain at least 3 positive values "
                    "for the L-curve corner search."
                )
        points = []
        for lm in lams:
            _, gamma_lm, resid_lm = _solve(w, z, tau, float(lm))
            rho = np.log(max(resid_lm, 1e-9))  # numerical floor
            eta = np.log(max(float(np.sum(gamma_lm**2)), 1e-12))
            points.append((rho, eta))
        pts = np.asarray(points, dtype=float)
        k = _lcurve_corner(pts)
        lam = float(lams[k])
        lams_out = np.asarray(lams, dtype=float)
        rho_out = pts[:, 0].copy()
        eta_out = pts[:, 1].copy()

    r_inf, gamma, residual_pct = _solve(w, z, tau, float(lam))
    return DRTResult(
        tau_s=tau,
        gamma_ohm=np.asarray(gamma, dtype=float),
        r_inf_ohm=float(r_inf),
        r_pol_ohm=float(np.sum(gamma)),
        lam=float(lam),
        residual_pct=float(residual_pct),
        n_points_used=int(f.size),
        lambda_grid=lams_out,
        lcurve_log_residual=rho_out,
        lcurve_log_solution_norm=eta_out,
    )


def find_drt_peaks(
    tau_s: npt.ArrayLike,
    gamma_ohm: npt.ArrayLike,
    *,
    min_fraction: float = 0.05,
) -> list[DRTPeak]:
    """Locate DRT peaks and integrate their resistances.

    A peak is a local maximum of ``gamma``; its resistance is the sum of
    ``gamma`` over the peak's watershed region, bounded by the minima
    between consecutive peaks (and the grid edges). Peaks whose integrated
    resistance falls below ``min_fraction`` of the total polarization
    resistance are discarded as regularization ripples.

    Parameters
    ----------
    tau_s : array_like
        Relaxation-time grid, seconds (as returned by
        :func:`compute_drt`).
    gamma_ohm : array_like
        Distributed resistance per grid element, ohm.
    min_fraction : float, optional
        Minimum peak fraction of the total polarization resistance
        (default 0.05).

    Returns
    -------
    list of DRTPeak
        Peaks in ascending relaxation-time order. Empty when the
        distribution carries no polarization resistance or has no local
        maxima.

    Raises
    ------
    ValueError
        If inputs are malformed.
    """
    tau = np.asarray(tau_s, dtype=float)
    g = np.asarray(gamma_ohm, dtype=float)
    if tau.ndim != 1 or g.ndim != 1 or tau.size != g.size:
        raise ValueError(
            "tau_s and gamma_ohm must be one-dimensional and of equal "
            f"length, got shapes {tau.shape} and {g.shape}."
        )
    if not (0.0 <= min_fraction < 1.0):
        raise ValueError(
            f"min_fraction must be in [0, 1), got {min_fraction}."
        )
    total = float(g.sum())
    if total <= 0:
        return []
    idx = [
        i
        for i in range(1, len(g) - 1)
        if g[i] >= g[i - 1] and g[i] > g[i + 1]
    ]
    if not idx:
        return []
    # Watershed boundaries at the minima between consecutive peaks.
    bounds = [0]
    for a, b in zip(idx[:-1], idx[1:]):
        bounds.append(a + int(np.argmin(g[a : b + 1])))
    bounds.append(len(g))
    peaks: list[DRTPeak] = []
    for k, i in enumerate(idx):
        r = float(g[bounds[k] : bounds[k + 1]].sum())
        if r >= min_fraction * total:
            peaks.append(
                DRTPeak(
                    tau_s=float(tau[i]),
                    resistance_ohm=r,
                    fraction=r / total,
                )
            )
    return peaks


def region_resistance(
    tau_s: npt.ArrayLike,
    gamma_ohm: npt.ArrayLike,
    *,
    tau_min_s: float,
    tau_max_s: float,
) -> float:
    """Integrated DRT resistance within a relaxation-time window.

    Sums ``gamma`` over grid points with ``tau_min_s <= tau <= tau_max_s``.
    Use this to attribute polarization resistance to a caller-defined
    relaxation-time region (e.g. separating a fast and a slow process);
    the window boundaries depend on the system under study and must come
    from the caller.

    Parameters
    ----------
    tau_s : array_like
        Relaxation-time grid, seconds.
    gamma_ohm : array_like
        Distributed resistance per grid element, ohm.
    tau_min_s, tau_max_s : float
        Relaxation-time window, seconds (``0 < tau_min_s < tau_max_s``).

    Returns
    -------
    float
        Summed resistance within the window, ohm (0.0 when the window
        contains grid points but no distributed resistance).

    Raises
    ------
    ValueError
        If inputs are malformed or the window does not overlap the grid.
    """
    tau = np.asarray(tau_s, dtype=float)
    g = np.asarray(gamma_ohm, dtype=float)
    if tau.ndim != 1 or g.ndim != 1 or tau.size != g.size:
        raise ValueError(
            "tau_s and gamma_ohm must be one-dimensional and of equal "
            f"length, got shapes {tau.shape} and {g.shape}."
        )
    if not (tau_min_s > 0 and tau_max_s > 0 and tau_min_s < tau_max_s):
        raise ValueError(
            "Relaxation-time window must satisfy 0 < tau_min_s < "
            f"tau_max_s, got tau_min_s={tau_min_s}, tau_max_s={tau_max_s}."
        )
    mask = (tau >= tau_min_s) & (tau <= tau_max_s)
    if not mask.any():
        raise ValueError(
            f"Window [{tau_min_s:g}, {tau_max_s:g}] s does not overlap "
            f"the tau grid [{tau.min():g}, {tau.max():g}] s. Choose a "
            "window inside the grid (or extend the grid via "
            "extension_decades)."
        )
    return float(g[mask].sum())
