"""Readers for BioLogic (EC-Lab) tab-separated text exports.

This module provides:

* :func:`read_biologic_txt` — a robust loader for tab-separated, latin1-encoded
  EC-Lab ``.txt``/``.mpt``-style exports with automatic header-row detection,
  numeric validation, and derived capacity columns.
* :func:`coulomb_counted_capacity_mah` — cumulative ``|I|*dt`` charge
  integration, the preferred capacity measure (see note below).
* :func:`resolve_ewe_column` — exact-match pinning of the DC potential column
  ``<Ewe>/V``, which a naive keyword search would confuse with the AC modulus
  column ``|Ewe|/V``.
* :func:`parse_filename_tail` — parser for the common
  ``..._<NN>_<TYPE>_<Cxx>.txt`` export-name convention
  (sequence number, technique, instrument channel).

Capacity preference
-------------------
The instrument accumulator column ``(Q-Qo)/mA.h`` can reset or otherwise
misbehave mid-file in some exports. This module therefore always computes a
coulomb-counted capacity (``cap_mah``, and ``cap_mah_cm2`` normalised by the
*required* electrode area) from ``time/s`` and ``<I>/mA``, and exposes the raw
instrument accumulator separately (``cap_q_mah`` / ``cap_q_mah_cm2``) for
cross-checking only. **Prefer the coulomb-counted columns.**

All failure modes raise :class:`ValueError` or :class:`FileNotFoundError` with
actionable messages; there are no silent fallbacks.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Canonical EC-Lab column names (exact strings as written by the instrument).
# --------------------------------------------------------------------------
COL_TIME = "time/s"
COL_CURRENT = "<I>/mA"
COL_EWE = "<Ewe>/V"           # DC (cycle-averaged) working-electrode potential
COL_EWE_MODULUS = "|Ewe|/V"   # AC potential modulus -- NOT a DC potential
COL_Q = "(Q-Qo)/mA.h"         # instrument charge accumulator (can reset)
COL_FREQ = "freq/Hz"
COL_RE_Z = "Re(Z)/Ohm"
COL_NEG_IM_Z = "-Im(Z)/Ohm"

#: Columns coerced to numeric (hard error on failure) when present.
_NUMERIC_COLUMNS: tuple[str, ...] = (
    COL_TIME, COL_CURRENT, COL_EWE, COL_EWE_MODULUS,
    COL_Q, COL_FREQ, COL_RE_Z, COL_NEG_IM_Z,
)

#: Tokens whose presence identifies the column-header line of an export.
_HEADER_TOKENS: tuple[str, ...] = (
    COL_TIME, COL_FREQ, COL_CURRENT, COL_EWE, COL_EWE_MODULUS,
    COL_RE_Z, COL_NEG_IM_Z,
)

_NB_HEADER_RE = re.compile(r"Nb header lines\s*:\s*(\d+)")
_MAX_HEADER_SCAN = 500


# --------------------------------------------------------------------------
# Filename-tail parsing
# --------------------------------------------------------------------------
_TAIL_RE = re.compile(
    r"_(?P<sequence>\d+)_(?P<technique>[A-Za-z][A-Za-z0-9]*)_(?P<channel>C\d+)"
    r"\.txt\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilenameTail:
    """The trailing ``_<NN>_<TYPE>_<Cxx>.txt`` tokens of an export filename.

    Attributes
    ----------
    sequence : int or None
        Position of the technique in the acquisition program (``01``, ``02``,
        ...). ``None`` when the filename does not match the convention.
    technique : str or None
        Technique token, e.g. ``"GEIS"``, ``"PEIS"``, ``"GCPL"``, ``"OCV"``.
        ``None`` when the filename does not match the convention.
    channel : str or None
        Instrument channel slot, e.g. ``"C11"``. This identifies a
        potentiostat slot, not a sample. ``None`` when the filename does not
        match the convention.
    """

    sequence: int | None
    technique: str | None
    channel: str | None

    @property
    def is_recognized(self) -> bool:
        """True when the filename matched the tail convention."""
        return self.sequence is not None


def parse_filename_tail(name: str | os.PathLike[str]) -> FilenameTail:
    """Parse the trailing ``_<NN>_<TYPE>_<Cxx>.txt`` tokens of a filename.

    Many EC-Lab acquisition programs export files whose last three
    underscore-separated tokens are (1) the sequence number of the technique
    in the program, (2) the technique name, and (3) the instrument channel.
    Earlier tokens are free-form metadata and are deliberately ignored here.

    Parameters
    ----------
    name : str or os.PathLike
        Filename or path. Only the basename is inspected.

    Returns
    -------
    FilenameTail
        Parsed fields. If the basename does not match the convention, a
        :class:`FilenameTail` with **all fields None** is returned — the
        parser never guesses.

    Examples
    --------
    >>> parse_filename_tail("run_A_03_GEIS_C11.txt")
    FilenameTail(sequence=3, technique='GEIS', channel='C11')
    >>> parse_filename_tail("notes.txt").is_recognized
    False
    """
    basename = Path(name).name
    match = _TAIL_RE.search(basename)
    if match is None:
        return FilenameTail(sequence=None, technique=None, channel=None)
    return FilenameTail(
        sequence=int(match.group("sequence")),
        technique=match.group("technique"),
        channel=match.group("channel"),
    )


# --------------------------------------------------------------------------
# Column pinning
# --------------------------------------------------------------------------
def resolve_ewe_column(columns: "pd.Index | list[str] | tuple[str, ...]") -> str:
    """Return the DC potential column name, pinned exactly to ``<Ewe>/V``.

    EC-Lab EIS exports may contain both ``<Ewe>/V`` (the DC, cycle-averaged
    working-electrode potential) and ``|Ewe|/V`` (the AC excitation modulus).
    A keyword search for ``"ewe"`` + ``"v"`` matches both — this function
    exists to avoid that trap: it only ever returns the exact string
    ``<Ewe>/V``.

    Parameters
    ----------
    columns : sequence of str
        Column names of a loaded export (e.g. ``df.columns``).

    Returns
    -------
    str
        Always the exact string ``"<Ewe>/V"``.

    Raises
    ------
    ValueError
        If ``<Ewe>/V`` is absent. The message distinguishes the case where
        only ``|Ewe|/V`` exists, because the AC modulus is *not* a valid
        substitute for the DC potential.
    """
    cols = list(columns)
    if COL_EWE in cols:
        return COL_EWE
    if COL_EWE_MODULUS in cols:
        raise ValueError(
            f"DC potential column {COL_EWE!r} not found; only the AC modulus "
            f"column {COL_EWE_MODULUS!r} is present. The modulus is the AC "
            "excitation amplitude and cannot be used as the cell potential. "
            "Re-export the data with the <Ewe> variable included."
        )
    raise ValueError(
        f"DC potential column {COL_EWE!r} not found. Available columns: "
        f"{cols!r}. Re-export the data with the <Ewe> variable included."
    )


# --------------------------------------------------------------------------
# Coulomb counting
# --------------------------------------------------------------------------
def coulomb_counted_capacity_mah(
    time_s: "np.ndarray | pd.Series | list[float]",
    current_ma: "np.ndarray | pd.Series | list[float]",
) -> np.ndarray:
    """Cumulative coulomb-counted capacity, ``cumsum(|I| * dt) / 3600`` (mAh).

    This is the preferred capacity measure: it depends only on the measured
    current and timestamps, whereas the instrument accumulator column
    ``(Q-Qo)/mA.h`` can reset mid-file in some exports.

    Parameters
    ----------
    time_s : array_like of float
        Timestamps in seconds. Must be 1-D, finite, and non-decreasing.
    current_ma : array_like of float
        Current in mA, same length as ``time_s``. Must be finite. The sign
        is discarded (``|I|``): the result is a cumulative *throughput*.

    Returns
    -------
    numpy.ndarray
        Cumulative capacity in mAh, same length as the inputs. The first
        element is 0 (no interval precedes the first sample).

    Raises
    ------
    ValueError
        If the inputs are empty, of mismatched length, non-finite, or if the
        time vector decreases anywhere (a symptom of concatenated or corrupt
        exports — split those into per-step files before loading).
    """
    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_ma, dtype=float)
    if t.ndim != 1 or i.ndim != 1:
        raise ValueError(
            f"time_s and current_ma must be 1-D; got shapes {t.shape} and {i.shape}."
        )
    if t.size == 0:
        raise ValueError("time_s is empty; cannot integrate capacity.")
    if t.size != i.size:
        raise ValueError(
            f"time_s (n={t.size}) and current_ma (n={i.size}) differ in length."
        )
    if not np.all(np.isfinite(t)):
        raise ValueError(
            f"time_s contains non-finite values (first at row "
            f"{int(np.flatnonzero(~np.isfinite(t))[0])})."
        )
    if not np.all(np.isfinite(i)):
        raise ValueError(
            f"current_ma contains non-finite values (first at row "
            f"{int(np.flatnonzero(~np.isfinite(i))[0])})."
        )
    dt = np.diff(t, prepend=t[0])
    if np.any(dt < 0):
        bad = int(np.flatnonzero(dt < 0)[0])
        raise ValueError(
            f"time_s decreases at row {bad} ({t[bad - 1]!r} -> {t[bad]!r}). "
            "The file may contain concatenated or corrupt segments; split it "
            "into monotonic-time pieces before loading."
        )
    return np.cumsum(np.abs(i) * dt) / 3600.0


# --------------------------------------------------------------------------
# Header detection
# --------------------------------------------------------------------------
def _locate_header_row(path: Path, encoding: str) -> int:
    """Return the 0-based line index of the column-header row.

    Strategy: (1) honour an EC-Lab ``Nb header lines : N`` marker when it is
    present near the top *and* line ``N`` (1-based) actually looks like a
    column header; otherwise (2) scan the first ``_MAX_HEADER_SCAN`` lines
    for the first line containing a known column token.

    Raises
    ------
    ValueError
        If the file is empty or no header line can be located.
    """
    lines: list[str] = []
    with open(path, encoding=encoding) as fh:
        for line_no, line in enumerate(fh):
            lines.append(line.rstrip("\r\n"))
            if line_no + 1 >= _MAX_HEADER_SCAN:
                break
    if not lines:
        raise ValueError(f"{path}: file is empty.")

    for line in lines[:5]:
        marker = _NB_HEADER_RE.search(line)
        if marker is not None:
            idx = int(marker.group(1)) - 1
            if 0 <= idx < len(lines) and any(
                token in lines[idx] for token in _HEADER_TOKENS
            ):
                return idx
            break  # marker present but inconsistent -> fall back to token scan

    for idx, line in enumerate(lines):
        if any(token in line for token in _HEADER_TOKENS):
            return idx

    raise ValueError(
        f"{path}: could not locate a column-header line within the first "
        f"{_MAX_HEADER_SCAN} lines. Expected a tab-separated line containing "
        f"at least one of {list(_HEADER_TOKENS)!r}. Verify this is a BioLogic "
        "text export and that the correct encoding was used."
    )


# --------------------------------------------------------------------------
# Main reader
# --------------------------------------------------------------------------
def read_biologic_txt(
    path: str | os.PathLike[str],
    *,
    area_cm2: float,
    sep: str = "\t",
    encoding: str = "latin1",
    decimal: str = ".",
) -> pd.DataFrame:
    """Load a BioLogic (EC-Lab) tab-separated text export into a DataFrame.

    Handles the common export quirks: free-form metadata lines before the
    column header (auto-detected), a trailing unnamed column produced by a
    trailing separator, latin1 encoding, and the broken instrument charge
    accumulator (see *Capacity preference* in the module docstring).

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``.txt`` export.
    area_cm2 : float
        Electrode area in cm**2, used to normalise capacity and current.
        **Required** — there is deliberately no default, because a wrong
        area silently corrupts every areal quantity downstream.
    sep : str, optional
        Field separator (default tab).
    encoding : str, optional
        Text encoding (default ``"latin1"``, the EC-Lab export default).
    decimal : str, optional
        Decimal separator (default ``"."``). Pass ``","`` for exports
        written under a comma-decimal locale.

    Returns
    -------
    pandas.DataFrame
        The export's columns (numeric columns validated/coerced), plus,
        when their source columns are present:

        * ``cap_mah`` — coulomb-counted cumulative ``|I|*dt`` capacity, mAh
          (**preferred**; needs ``time/s`` and ``<I>/mA``),
        * ``cap_mah_cm2`` — ``cap_mah / area_cm2``, mAh/cm**2,
        * ``j_ma_cm2`` — current density ``<I>/mA / area_cm2``, mA/cm**2,
        * ``cap_q_mah`` — the raw instrument accumulator ``(Q-Qo)/mA.h``,
          exposed for cross-checking only (it can reset mid-file),
        * ``cap_q_mah_cm2`` — ``cap_q_mah / area_cm2``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``area_cm2`` is not a finite positive number, no header line can
        be found, the file has no data rows, a known column fails numeric
        conversion, or the time vector decreases (see
        :func:`coulomb_counted_capacity_mah`).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{file_path}: no such file. If the data lives in a synced cloud "
            "folder, confirm the file has actually been downloaded locally."
        )
    if not isinstance(area_cm2, (int, float)) or isinstance(area_cm2, bool):
        raise ValueError(
            f"area_cm2 must be a positive number in cm^2; got {area_cm2!r}."
        )
    if not math.isfinite(area_cm2) or area_cm2 <= 0.0:
        raise ValueError(
            f"area_cm2 must be a finite positive number in cm^2; got {area_cm2!r}."
        )

    header_row = _locate_header_row(file_path, encoding=encoding)
    df = pd.read_csv(
        file_path,
        sep=sep,
        encoding=encoding,
        decimal=decimal,
        skiprows=header_row,
    )
    # EC-Lab exports often end each row with a trailing separator -> an
    # all-NaN 'Unnamed: N' column. Drop any such columns.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    if df.empty:
        raise ValueError(
            f"{file_path}: header found on line {header_row + 1} but no data "
            "rows follow. The export may be truncated."
        )

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{file_path}: column {col!r} contains non-numeric values "
                    f"({exc}). If the export uses comma decimals, pass "
                    "decimal=','."
                ) from exc

    if COL_TIME in df.columns and COL_CURRENT in df.columns:
        cap = coulomb_counted_capacity_mah(
            df[COL_TIME].to_numpy(dtype=float),
            df[COL_CURRENT].to_numpy(dtype=float),
        )
        df["cap_mah"] = cap
        df["cap_mah_cm2"] = cap / area_cm2
        df["j_ma_cm2"] = df[COL_CURRENT].to_numpy(dtype=float) / area_cm2

    if COL_Q in df.columns:
        df["cap_q_mah"] = df[COL_Q].to_numpy(dtype=float)
        df["cap_q_mah_cm2"] = df["cap_q_mah"] / area_cm2

    return df
