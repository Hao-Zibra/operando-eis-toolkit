"""Frequency-sweep segmentation for point-stream EIS exports (GEIS/PEIS).

Operando galvanostatic/potentiostatic EIS programs typically log many
consecutive frequency sweeps into a single point stream: each sweep descends
from a high start frequency (~MHz) toward a low end frequency (sub-Hz), and
the next sweep begins when the frequency *resets upward*. This module splits
such a stream into per-sweep index ranges and computes lightweight per-sweep
metadata (mean time, mean capacity when available, and a completeness flag
for sweeps truncated before reaching the end frequency — e.g. when a voltage
cutoff interrupted the program mid-sweep).

Reset detection is generic: a new sweep starts wherever the frequency rises
by more than ``reset_decades`` decades between consecutive rows. No
instrument-specific start/end frequencies are hard-coded; explicit markers
can be supplied to sharpen the completeness test.

All failure modes raise :class:`ValueError` with actionable messages; there
are no silent fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Default column names, matching BioLogic (EC-Lab) text exports.
DEFAULT_FREQ_COL = "freq/Hz"
DEFAULT_TIME_COL = "time/s"

#: Capacity columns tried (in order) when ``cap_col`` is not given. The first
#: two are produced by :func:`oeis_toolkit.io.biologic.read_biologic_txt`
#: (coulomb-counted; preferred); the last is the raw instrument accumulator.
_CAP_COL_CANDIDATES: tuple[str, ...] = ("cap_mah_cm2", "cap_mah", "(Q-Qo)/mA.h")


@dataclass(frozen=True)
class SweepMeta:
    """Index range and lightweight metadata for one frequency sweep.

    Attributes
    ----------
    start, stop : int
        Positional (``iloc``) row range of the sweep, ``stop`` exclusive.
    n_points : int
        Number of rows in the sweep (``stop - start``).
    f_first_hz, f_last_hz : float
        Frequency of the first and last row (acquisition order).
    f_max_hz, f_min_hz : float
        Extreme frequencies reached within the sweep.
    complete : bool
        True when the sweep descended to within ``tolerance_decades`` of the
        end-frequency reference (explicit ``f_end_hz`` if given, otherwise
        the lowest frequency observed anywhere in the stream). A False value
        flags a truncated sweep (e.g. cut short by a voltage limit).
    starts_at_top : bool or None
        Only computed when an explicit ``f_start_hz`` marker is supplied:
        True when the sweep's maximum frequency is within
        ``tolerance_decades`` of it. ``None`` when no marker was given.
    time_mean_s : float or None
        Mean of the time column over the sweep, or ``None`` if the column
        is absent.
    cap_mean : float or None
        Mean of the capacity column over the sweep, or ``None`` if no
        capacity column was found. Units follow ``cap_col``.
    cap_col : str or None
        Name of the capacity column actually used for ``cap_mean``.
    """

    start: int
    stop: int
    n_points: int
    f_first_hz: float
    f_last_hz: float
    f_max_hz: float
    f_min_hz: float
    complete: bool
    starts_at_top: bool | None
    time_mean_s: float | None
    cap_mean: float | None
    cap_col: str | None


def sweep_bounds(
    freq_hz: "np.ndarray | pd.Series | Sequence[float]",
    *,
    reset_decades: float = 2.0,
) -> list[tuple[int, int]]:
    """Split a frequency point stream into per-sweep index ranges.

    A new sweep starts at row ``i`` whenever
    ``log10(f[i]) - log10(f[i-1]) > reset_decades``, i.e. the frequency jumps
    back up toward the sweep start. Within a descending sweep consecutive
    steps are small (fractions of a decade), while the reset jump spans most
    of the sweep's range, so a threshold of a couple of decades separates the
    two robustly without hard-coding instrument frequencies.

    Parameters
    ----------
    freq_hz : array_like of float
        Frequency of each row, in acquisition order. Must be 1-D, finite
        and strictly positive.
    reset_decades : float, optional
        Minimum upward jump, in decades, that marks the start of a new
        sweep. Default 2.0. Must be positive. Lower it if your sweeps span
        little more than two decades; raise it if genuine within-sweep
        upward steps exceed it.

    Returns
    -------
    list of (int, int)
        Half-open positional ranges ``(start, stop)``, one per sweep, in
        order, covering all rows. Empty list for empty input.

    Raises
    ------
    ValueError
        If the input is not 1-D, contains non-finite or non-positive
        frequencies, or if ``reset_decades`` is not positive.
    """
    if not (isinstance(reset_decades, (int, float)) and reset_decades > 0):
        raise ValueError(
            f"reset_decades must be a positive number; got {reset_decades!r}."
        )
    f = np.asarray(freq_hz, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"freq_hz must be 1-D; got shape {f.shape}.")
    if f.size == 0:
        return []
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        bad = int(np.flatnonzero(~(np.isfinite(f) & (f > 0)))[0])
        raise ValueError(
            f"freq_hz must be finite and strictly positive; offending value "
            f"{f[bad]!r} at row {bad}. Remove non-EIS rows before segmenting."
        )
    logf = np.log10(f)
    starts = [0] + [
        i for i in range(1, f.size) if logf[i] - logf[i - 1] > reset_decades
    ]
    edges = starts + [f.size]
    return list(zip(edges[:-1], edges[1:]))


def segment_sweeps(
    df: pd.DataFrame,
    *,
    freq_col: str = DEFAULT_FREQ_COL,
    time_col: str = DEFAULT_TIME_COL,
    cap_col: str | None = None,
    reset_decades: float = 2.0,
    f_start_hz: float | None = None,
    f_end_hz: float | None = None,
    tolerance_decades: float = 1.0,
) -> list[SweepMeta]:
    """Segment an EIS point stream and summarise each sweep.

    Parameters
    ----------
    df : pandas.DataFrame
        Point stream containing at least ``freq_col``. Rows must be in
        acquisition order; ranges in the result are positional (``iloc``).
    freq_col : str, optional
        Frequency column name (default ``"freq/Hz"``). Missing -> error.
    time_col : str, optional
        Time column name (default ``"time/s"``). If absent,
        ``time_mean_s`` is ``None`` (time metadata is optional).
    cap_col : str or None, optional
        Capacity column for ``cap_mean``. If ``None`` (default) the first
        available of ``cap_mah_cm2``, ``cap_mah``, ``(Q-Qo)/mA.h`` is used;
        if none exist, ``cap_mean`` is ``None``. If given explicitly, the
        column must exist (error otherwise).
    reset_decades : float, optional
        Passed to :func:`sweep_bounds`.
    f_start_hz, f_end_hz : float or None, optional
        Optional explicit sweep start/end frequency markers. When
        ``f_end_hz`` is ``None`` the lowest frequency observed anywhere in
        the stream is used as the end reference — note that with this
        inferred reference the deepest sweep is complete by construction,
        so supply ``f_end_hz`` when an absolute completeness test matters.
        ``f_start_hz`` only enables the ``starts_at_top`` flag.
    tolerance_decades : float, optional
        A sweep is ``complete`` when ``f_min_hz`` lies within this many
        decades above the end reference (default 1.0). Must be >= 0.

    Returns
    -------
    list of SweepMeta
        One entry per detected sweep, in order. Empty list if ``df`` has
        no rows.

    Raises
    ------
    ValueError
        If ``freq_col`` is missing, an explicitly requested ``cap_col`` is
        missing, marker frequencies are non-positive, ``tolerance_decades``
        is negative, or the frequency data is invalid (see
        :func:`sweep_bounds`).
    """
    if freq_col not in df.columns:
        raise ValueError(
            f"Frequency column {freq_col!r} not found. Available columns: "
            f"{list(df.columns)!r}. This function expects an EIS point "
            "stream; pass freq_col=... if your export names it differently."
        )
    if not (isinstance(tolerance_decades, (int, float)) and tolerance_decades >= 0):
        raise ValueError(
            f"tolerance_decades must be a non-negative number; got "
            f"{tolerance_decades!r}."
        )
    for marker_name, marker in (("f_start_hz", f_start_hz), ("f_end_hz", f_end_hz)):
        if marker is not None and not (
            isinstance(marker, (int, float)) and np.isfinite(marker) and marker > 0
        ):
            raise ValueError(
                f"{marker_name} must be a finite positive frequency in Hz; "
                f"got {marker!r}."
            )
    if cap_col is not None and cap_col not in df.columns:
        raise ValueError(
            f"Requested capacity column {cap_col!r} not found. Available "
            f"columns: {list(df.columns)!r}. Pass cap_col=None to "
            "auto-detect or omit capacity metadata."
        )

    f = df[freq_col].to_numpy(dtype=float)
    bounds = sweep_bounds(f, reset_decades=reset_decades)
    if not bounds:
        return []

    end_ref = float(f.min()) if f_end_hz is None else float(f_end_hz)
    complete_ceiling = end_ref * 10.0 ** tolerance_decades
    top_floor = (
        None if f_start_hz is None else float(f_start_hz) * 10.0 ** -tolerance_decades
    )

    effective_cap_col: str | None = cap_col
    if effective_cap_col is None:
        for candidate in _CAP_COL_CANDIDATES:
            if candidate in df.columns:
                effective_cap_col = candidate
                break

    time_values = (
        df[time_col].to_numpy(dtype=float) if time_col in df.columns else None
    )
    cap_values = (
        df[effective_cap_col].to_numpy(dtype=float)
        if effective_cap_col is not None
        else None
    )

    sweeps: list[SweepMeta] = []
    for start, stop in bounds:
        f_blk = f[start:stop]
        f_min = float(f_blk.min())
        f_max = float(f_blk.max())
        sweeps.append(
            SweepMeta(
                start=start,
                stop=stop,
                n_points=stop - start,
                f_first_hz=float(f_blk[0]),
                f_last_hz=float(f_blk[-1]),
                f_max_hz=f_max,
                f_min_hz=f_min,
                complete=bool(f_min <= complete_ceiling),
                starts_at_top=(
                    None if top_floor is None else bool(f_max >= top_floor)
                ),
                time_mean_s=(
                    None
                    if time_values is None
                    else float(np.mean(time_values[start:stop]))
                ),
                cap_mean=(
                    None
                    if cap_values is None
                    else float(np.mean(cap_values[start:stop]))
                ),
                cap_col=effective_cap_col,
            )
        )
    return sweeps


def sweep_frames(
    df: pd.DataFrame,
    sweeps: "Iterable[SweepMeta | tuple[int, int]]",
) -> list[pd.DataFrame]:
    """Slice a point stream into one DataFrame per sweep.

    Parameters
    ----------
    df : pandas.DataFrame
        The point stream that was segmented.
    sweeps : iterable of SweepMeta or (start, stop) tuples
        Output of :func:`segment_sweeps` or :func:`sweep_bounds`.

    Returns
    -------
    list of pandas.DataFrame
        Positional (``iloc``) slices of ``df``, one per sweep, in order.

    Raises
    ------
    ValueError
        If any range falls outside ``df``.
    """
    frames: list[pd.DataFrame] = []
    n = len(df)
    for item in sweeps:
        if isinstance(item, SweepMeta):
            start, stop = item.start, item.stop
        else:
            start, stop = item
        if not (0 <= start <= stop <= n):
            raise ValueError(
                f"Sweep range ({start}, {stop}) is outside the frame "
                f"(len={n}). Segment and slice the same DataFrame."
            )
        frames.append(df.iloc[start:stop])
    return frames
