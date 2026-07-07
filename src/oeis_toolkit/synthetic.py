"""Deterministic synthetic-data generators for examples and tests.

Everything here is self-contained (numpy/pandas only) and reproducible: each
generator takes a required ``rng`` argument (a :class:`numpy.random.Generator`
or an integer seed) and never touches global random state.

Generators
----------
* :func:`nyquist_dataset` — impedance spectra from an
  ``R_s + N x (R || CPE)`` equivalent circuit, with optional proportional
  Gaussian noise (the pure model is exposed as :func:`impedance_rs_rcpe`).
* :func:`weibull_censored_sample` — right-censored Weibull lifetime samples
  for reliability-analysis examples.
* :func:`capacity_fade_series` — linear or exponential capacity-fade curves
  with additive noise.
* :func:`geis_stream` — a multi-sweep GEIS-style point stream (concatenated
  descending frequency sweeps with upward resets, optionally a truncated
  final sweep) using BioLogic export column names, for testing sweep
  segmentation and file-reader round trips.

All argument validation raises :class:`ValueError` with actionable messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

#: One (R_ohm, Q, alpha) parallel R||CPE arc: Z_arc = R / (1 + R*Q*(j*w)**alpha).
ArcParams = tuple[float, float, float]


def _as_rng(rng: "np.random.Generator | int") -> np.random.Generator:
    """Normalise ``rng`` to a Generator (an int is treated as a seed)."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    raise ValueError(
        f"rng must be a numpy.random.Generator or an integer seed; got "
        f"{type(rng).__name__}."
    )


def _validate_freq(freq_hz: "np.ndarray | Sequence[float]") -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    if f.ndim != 1 or f.size == 0:
        raise ValueError(f"freq_hz must be a non-empty 1-D array; got shape {f.shape}.")
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("freq_hz must contain only finite, strictly positive values.")
    return f


def _validate_arcs(arcs: Sequence[ArcParams]) -> list[ArcParams]:
    validated: list[ArcParams] = []
    for k, arc in enumerate(arcs):
        if len(arc) != 3:
            raise ValueError(
                f"arcs[{k}] must be a (R_ohm, Q, alpha) triple; got {arc!r}."
            )
        r, q, alpha = (float(v) for v in arc)
        if r <= 0 or q <= 0:
            raise ValueError(
                f"arcs[{k}]: R_ohm and Q must be positive; got R={r!r}, Q={q!r}."
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"arcs[{k}]: alpha must be in (0, 1]; got {alpha!r}."
            )
        validated.append((r, q, alpha))
    return validated


# --------------------------------------------------------------------------
# (a) Nyquist data from R_s + N x (R || CPE)
# --------------------------------------------------------------------------
def impedance_rs_rcpe(
    freq_hz: "np.ndarray | Sequence[float]",
    r_s: float,
    arcs: Sequence[ArcParams],
) -> np.ndarray:
    """Complex impedance of ``R_s`` in series with N parallel R||CPE arcs.

    ``Z(w) = R_s + sum_k R_k / (1 + R_k * Q_k * (j*w)**alpha_k)`` with
    ``w = 2*pi*f``.

    Parameters
    ----------
    freq_hz : array_like of float
        Frequency grid in Hz (1-D, finite, positive).
    r_s : float
        Series (high-frequency) resistance in ohm, >= 0.
    arcs : sequence of (R_ohm, Q, alpha)
        Parameters of each parallel R||CPE arc. ``alpha`` in (0, 1]
        (``alpha = 1`` reduces the CPE to an ideal capacitor of value Q).

    Returns
    -------
    numpy.ndarray of complex
        Impedance at each frequency, same length as ``freq_hz``.

    Raises
    ------
    ValueError
        On invalid frequencies, ``r_s < 0``, or invalid arc parameters.
    """
    f = _validate_freq(freq_hz)
    if not (isinstance(r_s, (int, float)) and np.isfinite(r_s) and r_s >= 0):
        raise ValueError(f"r_s must be a finite non-negative resistance; got {r_s!r}.")
    w = 2.0 * np.pi * f
    z = np.full(f.shape, float(r_s), dtype=complex)
    for r, q, alpha in _validate_arcs(arcs):
        z += r / (1.0 + r * q * (1j * w) ** alpha)
    return z


def nyquist_dataset(
    freq_hz: "np.ndarray | Sequence[float]",
    *,
    r_s: float,
    arcs: Sequence[ArcParams],
    noise_frac: float = 0.0,
    rng: "np.random.Generator | int",
) -> pd.DataFrame:
    """Synthetic Nyquist dataset from known circuit parameters.

    Parameters
    ----------
    freq_hz : array_like of float
        Frequency grid in Hz.
    r_s : float
        Series resistance in ohm.
    arcs : sequence of (R_ohm, Q, alpha)
        R||CPE arcs (see :func:`impedance_rs_rcpe`).
    noise_frac : float, optional
        Standard deviation of the added Gaussian noise as a fraction of
        ``|Z|`` at each point, applied independently to the real and
        imaginary parts. Default 0.0 (noise-free). Must be >= 0.
    rng : numpy.random.Generator or int
        Random generator or integer seed (required, for determinism).

    Returns
    -------
    pandas.DataFrame
        Columns ``freq/Hz``, ``Re(Z)/Ohm``, ``-Im(Z)/Ohm`` (BioLogic export
        naming, so the result plugs directly into the EIS tooling).

    Raises
    ------
    ValueError
        On invalid parameters (see :func:`impedance_rs_rcpe`) or negative
        ``noise_frac``.
    """
    if not (isinstance(noise_frac, (int, float)) and noise_frac >= 0):
        raise ValueError(f"noise_frac must be >= 0; got {noise_frac!r}.")
    generator = _as_rng(rng)
    f = _validate_freq(freq_hz)
    z = impedance_rs_rcpe(f, r_s, arcs)
    if noise_frac > 0:
        scale = noise_frac * np.abs(z)
        z = z + scale * (
            generator.standard_normal(f.size)
            + 1j * generator.standard_normal(f.size)
        )
    return pd.DataFrame(
        {"freq/Hz": f, "Re(Z)/Ohm": z.real, "-Im(Z)/Ohm": -z.imag}
    )


# --------------------------------------------------------------------------
# (b) Censored Weibull lifetimes
# --------------------------------------------------------------------------
def weibull_censored_sample(
    *,
    beta: float,
    eta: float,
    n: int,
    censor_time: "float | np.ndarray | Sequence[float] | None" = None,
    rng: "np.random.Generator | int",
) -> pd.DataFrame:
    """Right-censored sample from a Weibull(beta, eta) lifetime distribution.

    Latent failure times are drawn as ``t = eta * W(beta)`` where ``W`` is
    numpy's standard Weibull. Each unit is observed until ``min(t, c)``
    where ``c`` is its censor time; units with ``t > c`` are right-censored.

    Parameters
    ----------
    beta : float
        Weibull shape parameter, > 0.
    eta : float
        Weibull scale (characteristic life), > 0. Units are whatever the
        caller's lifetime metric is (cycles, hours, throughput, ...).
    n : int
        Sample size, >= 1.
    censor_time : float, array_like of float, or None, optional
        Censoring rule: a scalar applies one fixed (Type-I) censor time to
        every unit; an array of length ``n`` gives per-unit censor times
        (e.g. staggered test stops); ``None`` (default) disables censoring
        (all units run to failure). Censor times must be positive.
    rng : numpy.random.Generator or int
        Random generator or integer seed (required, for determinism).

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (observed time, float) and ``event`` (int: 1 =
        failure observed, 0 = right-censored).

    Raises
    ------
    ValueError
        On non-positive ``beta``/``eta``, ``n < 1``, or invalid
        ``censor_time``.
    """
    if not (isinstance(beta, (int, float)) and np.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be a finite positive number; got {beta!r}.")
    if not (isinstance(eta, (int, float)) and np.isfinite(eta) and eta > 0):
        raise ValueError(f"eta must be a finite positive number; got {eta!r}.")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    generator = _as_rng(rng)

    t = float(eta) * generator.weibull(float(beta), int(n))
    if censor_time is None:
        return pd.DataFrame({"time": t, "event": np.ones(int(n), dtype=int)})

    c = np.asarray(censor_time, dtype=float)
    if c.ndim == 0:
        c = np.full(int(n), float(c))
    if c.shape != (int(n),):
        raise ValueError(
            f"censor_time must be a scalar or an array of length n={n}; got "
            f"shape {c.shape}."
        )
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("censor_time values must be finite and positive.")
    observed = np.minimum(t, c)
    event = (t <= c).astype(int)
    return pd.DataFrame({"time": observed, "event": event})


# --------------------------------------------------------------------------
# (c) Capacity-fade series
# --------------------------------------------------------------------------
def capacity_fade_series(
    *,
    n_cycles: int,
    q0: float,
    fade_rate: float,
    model: str = "linear",
    noise_sd: float = 0.0,
    rng: "np.random.Generator | int",
) -> pd.DataFrame:
    """Synthetic capacity-vs-cycle series with linear or exponential fade.

    * ``model="linear"``: ``q_k = q0 * (1 - fade_rate * k)``
    * ``model="exp"``:    ``q_k = q0 * exp(-fade_rate * k)``

    for cycle numbers ``k = 1 .. n_cycles``, plus additive Gaussian noise of
    standard deviation ``noise_sd`` (same units as ``q0``). Note the linear
    model can go negative for large ``fade_rate * n_cycles``; values are
    returned as-is (no clipping) so tests see the pure model.

    Parameters
    ----------
    n_cycles : int
        Number of cycles, >= 1.
    q0 : float
        Initial capacity (any unit, e.g. mAh or mAh/cm**2), > 0.
    fade_rate : float
        Fractional fade per cycle, >= 0.
    model : {"linear", "exp"}, optional
        Fade law. Default ``"linear"``.
    noise_sd : float, optional
        Additive Gaussian noise standard deviation. Default 0.0. Must be
        >= 0.
    rng : numpy.random.Generator or int
        Random generator or integer seed (required, for determinism).

    Returns
    -------
    pandas.DataFrame
        Columns ``cycle`` (int, 1-based) and ``capacity`` (float, units of
        ``q0``).

    Raises
    ------
    ValueError
        On invalid ``n_cycles``, ``q0``, ``fade_rate``, ``noise_sd``, or an
        unknown ``model``.
    """
    if not isinstance(n_cycles, (int, np.integer)) or isinstance(n_cycles, bool) \
            or n_cycles < 1:
        raise ValueError(f"n_cycles must be an integer >= 1; got {n_cycles!r}.")
    if not (isinstance(q0, (int, float)) and np.isfinite(q0) and q0 > 0):
        raise ValueError(f"q0 must be a finite positive capacity; got {q0!r}.")
    if not (isinstance(fade_rate, (int, float)) and np.isfinite(fade_rate)
            and fade_rate >= 0):
        raise ValueError(f"fade_rate must be a finite number >= 0; got {fade_rate!r}.")
    if not (isinstance(noise_sd, (int, float)) and np.isfinite(noise_sd)
            and noise_sd >= 0):
        raise ValueError(f"noise_sd must be a finite number >= 0; got {noise_sd!r}.")
    generator = _as_rng(rng)

    cycles = np.arange(1, int(n_cycles) + 1)
    if model == "linear":
        q = float(q0) * (1.0 - float(fade_rate) * cycles)
    elif model == "exp":
        q = float(q0) * np.exp(-float(fade_rate) * cycles)
    else:
        raise ValueError(
            f"model must be 'linear' or 'exp'; got {model!r}."
        )
    if noise_sd > 0:
        q = q + float(noise_sd) * generator.standard_normal(cycles.size)
    return pd.DataFrame({"cycle": cycles, "capacity": q})


# --------------------------------------------------------------------------
# (d) Multi-sweep GEIS-style point stream
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticGeisStream:
    """A synthetic GEIS point stream plus its ground truth.

    Attributes
    ----------
    frame : pandas.DataFrame
        Point stream with BioLogic export column names (``freq/Hz``,
        ``Re(Z)/Ohm``, ``-Im(Z)/Ohm``, ``time/s``, ``<I>/mA``, ``<Ewe>/V``,
        plus optional ``(Q-Qo)/mA.h`` and ``|Ewe|/V``).
    sweep_bounds : list of (int, int)
        Ground-truth half-open positional index range of each sweep, for
        checking segmentation output.
    truncated_last : bool
        True when the final sweep was truncated (did not reach the end
        frequency).
    """

    frame: pd.DataFrame
    sweep_bounds: list[tuple[int, int]]
    truncated_last: bool


def geis_stream(
    *,
    n_sweeps: int,
    points_per_sweep: int,
    f_start_hz: float = 1.0e6,
    f_end_hz: float = 0.1,
    truncate_last_at_hz: float | None = None,
    current_ma: float = 0.1,
    r_s: float = 50.0,
    arcs: Sequence[ArcParams] = ((200.0, 1.0e-7, 0.9),),
    noise_frac: float = 0.0,
    settle_s: float = 0.5,
    cycles_per_point: float = 2.0,
    v_offset_v: float = 0.05,
    v_drift_v_per_s: float = 0.0,
    include_broken_q: bool = False,
    include_ewe_modulus: bool = False,
    rng: "np.random.Generator | int",
) -> SyntheticGeisStream:
    """Synthetic multi-sweep GEIS point stream for segmentation/reader tests.

    Builds ``n_sweeps`` frequency sweeps, each descending log-uniformly from
    ``f_start_hz`` to ``f_end_hz`` over ``points_per_sweep`` rows; the
    frequency then resets upward at each sweep boundary, exactly as in a
    looped operando GEIS program. Optionally the final sweep is truncated at
    ``truncate_last_at_hz`` (mimicking a run cut short mid-sweep). Timestamps
    accumulate realistically (low-frequency points take longer:
    ``dt = settle_s + cycles_per_point / f``), current is constant, and
    impedance columns come from :func:`impedance_rs_rcpe`.

    For reset detection with the default two-decade threshold of
    :func:`oeis_toolkit.eis.segmentation.sweep_bounds`, keep the sweep span
    ``log10(f_start_hz / max(f_end_hz, truncate_last_at_hz))`` above ~2.5
    decades.

    Parameters
    ----------
    n_sweeps : int
        Number of sweeps, >= 1.
    points_per_sweep : int
        Rows per full sweep, >= 2.
    f_start_hz, f_end_hz : float, optional
        Sweep start (high) and end (low) frequencies in Hz; defaults span
        1 MHz -> 0.1 Hz. Must satisfy ``f_start_hz > f_end_hz > 0``.
    truncate_last_at_hz : float or None, optional
        If given, the final sweep keeps only rows with frequency >= this
        value (must lie strictly between ``f_end_hz`` and ``f_start_hz``).
        ``None`` (default) leaves all sweeps complete.
    current_ma : float, optional
        Constant applied current in mA (sign preserved in ``<I>/mA``).
    r_s : float, optional
        Series resistance of the synthetic circuit, ohm.
    arcs : sequence of (R_ohm, Q, alpha), optional
        R||CPE arcs of the synthetic circuit.
    noise_frac : float, optional
        Proportional Gaussian noise on the impedance (see
        :func:`nyquist_dataset`). Default 0.0.
    settle_s : float, optional
        Fixed per-point acquisition overhead, seconds. Must be >= 0.
    cycles_per_point : float, optional
        AC periods integrated per point; sets the frequency-dependent part
        of the per-point duration. Must be >= 0.
    v_offset_v, v_drift_v_per_s : float, optional
        ``<Ewe>/V`` is generated as ``v_offset_v + v_drift_v_per_s * t``.
    include_broken_q : bool, optional
        If True, add a ``(Q-Qo)/mA.h`` column that **resets to zero at every
        sweep boundary**, reproducing the instrument-accumulator failure
        mode that coulomb counting guards against. Default False.
    include_ewe_modulus : bool, optional
        If True, add a constant ``|Ewe|/V`` column (0.010 V placeholder AC
        amplitude) so readers can be tested against the
        ``<Ewe>/V`` vs ``|Ewe|/V`` column-pinning trap. Default False.
    rng : numpy.random.Generator or int
        Random generator or integer seed (required, for determinism).

    Returns
    -------
    SyntheticGeisStream
        The frame plus ground-truth sweep bounds and truncation flag.

    Raises
    ------
    ValueError
        On invalid sweep counts, frequency ordering, truncation marker, or
        circuit/noise/timing parameters.
    """
    if not isinstance(n_sweeps, (int, np.integer)) or isinstance(n_sweeps, bool) \
            or n_sweeps < 1:
        raise ValueError(f"n_sweeps must be an integer >= 1; got {n_sweeps!r}.")
    if not isinstance(points_per_sweep, (int, np.integer)) \
            or isinstance(points_per_sweep, bool) or points_per_sweep < 2:
        raise ValueError(
            f"points_per_sweep must be an integer >= 2; got {points_per_sweep!r}."
        )
    for name, value in (("f_start_hz", f_start_hz), ("f_end_hz", f_end_hz)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a finite positive frequency; got {value!r}.")
    if not f_start_hz > f_end_hz:
        raise ValueError(
            f"f_start_hz ({f_start_hz!r}) must exceed f_end_hz ({f_end_hz!r}): "
            "sweeps descend in frequency."
        )
    if truncate_last_at_hz is not None and not (
        isinstance(truncate_last_at_hz, (int, float))
        and f_end_hz < truncate_last_at_hz < f_start_hz
    ):
        raise ValueError(
            f"truncate_last_at_hz must lie strictly between f_end_hz "
            f"({f_end_hz!r}) and f_start_hz ({f_start_hz!r}); got "
            f"{truncate_last_at_hz!r}."
        )
    for name, value in (("settle_s", settle_s), ("cycles_per_point", cycles_per_point)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be a finite number >= 0; got {value!r}.")
    if settle_s == 0 and cycles_per_point == 0:
        raise ValueError(
            "settle_s and cycles_per_point cannot both be zero: timestamps "
            "would not advance."
        )
    if not (isinstance(noise_frac, (int, float)) and noise_frac >= 0):
        raise ValueError(f"noise_frac must be >= 0; got {noise_frac!r}.")
    generator = _as_rng(rng)

    full_freq = np.logspace(
        np.log10(float(f_start_hz)), np.log10(float(f_end_hz)),
        int(points_per_sweep),
    )

    freq_blocks: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    cursor = 0
    truncated_last = False
    for sweep_idx in range(int(n_sweeps)):
        block = full_freq
        if truncate_last_at_hz is not None and sweep_idx == int(n_sweeps) - 1:
            block = full_freq[full_freq >= float(truncate_last_at_hz)]
            truncated_last = block.size < full_freq.size
        freq_blocks.append(block)
        bounds.append((cursor, cursor + block.size))
        cursor += block.size

    freq = np.concatenate(freq_blocks)
    dt = float(settle_s) + float(cycles_per_point) / freq
    time_s = np.cumsum(dt)

    z = impedance_rs_rcpe(freq, r_s, arcs)
    if noise_frac > 0:
        scale = noise_frac * np.abs(z)
        z = z + scale * (
            generator.standard_normal(freq.size)
            + 1j * generator.standard_normal(freq.size)
        )

    data: dict[str, np.ndarray] = {
        "freq/Hz": freq,
        "Re(Z)/Ohm": z.real,
        "-Im(Z)/Ohm": -z.imag,
        "time/s": time_s,
        "<I>/mA": np.full(freq.size, float(current_ma)),
        "<Ewe>/V": float(v_offset_v) + float(v_drift_v_per_s) * time_s,
    }
    if include_ewe_modulus:
        data["|Ewe|/V"] = np.full(freq.size, 0.010)
    if include_broken_q:
        # Signed instrument-style accumulator that resets at every sweep
        # boundary — deliberately broken, to exercise coulomb counting.
        q = np.empty(freq.size)
        for start, stop in bounds:
            block_dt = np.diff(time_s[start:stop], prepend=time_s[start])
            q[start:stop] = np.cumsum(float(current_ma) * block_dt) / 3600.0
        data["(Q-Qo)/mA.h"] = q

    return SyntheticGeisStream(
        frame=pd.DataFrame(data),
        sweep_bounds=bounds,
        truncated_last=truncated_last,
    )
