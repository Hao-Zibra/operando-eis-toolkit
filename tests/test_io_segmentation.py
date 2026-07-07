"""Tests for the BioLogic reader (oeis_toolkit.io.biologic) and the EIS
sweep segmentation (oeis_toolkit.eis.segmentation), using the deterministic
generators in oeis_toolkit.synthetic.

Run with:  pytest tests/test_io_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# src-layout import without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oeis_toolkit.io.biologic import (  # noqa: E402
    COL_EWE,
    COL_EWE_MODULUS,
    coulomb_counted_capacity_mah,
    parse_filename_tail,
    read_biologic_txt,
    resolve_ewe_column,
)
from oeis_toolkit.eis.segmentation import (  # noqa: E402
    segment_sweeps,
    sweep_bounds,
    sweep_frames,
)
from oeis_toolkit.synthetic import geis_stream  # noqa: E402


# ---------------------------------------------------------------- helpers
def _write_export(path: Path, df: pd.DataFrame, preamble: str = "") -> Path:
    """Write a DataFrame as a tab-separated latin1 export, with optional
    metadata lines before the column header (as real exports have)."""
    text = preamble + df.to_csv(sep="\t", index=False)
    path.write_text(text, encoding="latin1")
    return path


# ------------------------------------------------------- filename-tail parser
def test_parse_tail_standard() -> None:
    tail = parse_filename_tail("sampleA_run1_03_GEIS_C11.txt")
    assert tail.sequence == 3
    assert tail.technique == "GEIS"
    assert tail.channel == "C11"
    assert tail.is_recognized


def test_parse_tail_many_leading_tokens_and_path() -> None:
    tail = parse_filename_tail(Path("some_dir") / "a_b_c_d_12_PEIS_C03.txt")
    assert (tail.sequence, tail.technique, tail.channel) == (12, "PEIS", "C03")


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",                # no tail at all
        "sample_GEIS_C11.txt",      # missing sequence number
        "sample_01_GEIS.txt",       # missing channel
        "sample_01_GEIS_C11.csv",   # wrong extension
        "sample_01_2EIS_C11.txt",   # technique must start with a letter
        "sample_01_GEIS_11.txt",    # channel must be C<digits>
    ],
)
def test_parse_tail_malformed_returns_all_none(name: str) -> None:
    tail = parse_filename_tail(name)
    assert tail.sequence is None
    assert tail.technique is None
    assert tail.channel is None
    assert not tail.is_recognized


# ------------------------------------------------------------ column pinning
def test_resolve_ewe_prefers_exact_dc_column() -> None:
    # A naive "ewe"+"v" keyword match would hit both columns; the resolver
    # must pin the DC potential exactly.
    df = pd.DataFrame({COL_EWE_MODULUS: [0.01], COL_EWE: [0.1], "time/s": [0.0]})
    assert resolve_ewe_column(df.columns) == COL_EWE


def test_resolve_ewe_rejects_modulus_only() -> None:
    with pytest.raises(ValueError, match="AC"):
        resolve_ewe_column([COL_EWE_MODULUS, "time/s"])


def test_resolve_ewe_rejects_absent() -> None:
    with pytest.raises(ValueError):
        resolve_ewe_column(["time/s", "<I>/mA"])


# ------------------------------------------------- coulomb-counted capacity
def test_coulomb_capacity_preferred_over_broken_q(tmp_path: Path) -> None:
    t = np.arange(0.0, 3601.0, 60.0)          # 0 .. 3600 s
    i = np.full(t.size, 1.2)                  # constant 1.2 mA
    q_broken = (1.2 * t / 3600.0) % 0.5       # accumulator that resets
    df = pd.DataFrame(
        {
            "time/s": t,
            "<I>/mA": i,
            "(Q-Qo)/mA.h": q_broken,
            COL_EWE: 0.05 + 1e-6 * t,
            COL_EWE_MODULUS: np.full(t.size, 0.01),
        }
    )
    path = _write_export(
        tmp_path / "cap_case_01_GEIS_C01.txt",
        df,
        preamble="ASCII EXPORT\nAcquisition started on : 2026-01-01\n",
    )

    loaded = read_biologic_txt(path, area_cm2=1.5)

    # Coulomb-counted capacity: monotone, correct total (1.2 mA for 1 h).
    cap = loaded["cap_mah"].to_numpy()
    assert np.all(np.diff(cap) >= 0)
    assert cap[0] == pytest.approx(0.0)
    assert cap[-1] == pytest.approx(1.2, rel=1e-9)
    assert loaded["cap_mah_cm2"].to_numpy()[-1] == pytest.approx(1.2 / 1.5)
    assert loaded["j_ma_cm2"].to_numpy()[0] == pytest.approx(1.2 / 1.5)

    # The instrument accumulator is exposed unmodified and is visibly broken.
    cap_q = loaded["cap_q_mah"].to_numpy()
    assert np.diff(cap_q).min() < 0
    assert "cap_q_mah_cm2" in loaded.columns

    # Column pinning survives the round trip (both Ewe columns present).
    assert resolve_ewe_column(loaded.columns) == COL_EWE


def test_coulomb_capacity_rejects_decreasing_time() -> None:
    with pytest.raises(ValueError, match="decreases"):
        coulomb_counted_capacity_mah([0.0, 10.0, 5.0], [1.0, 1.0, 1.0])


# ------------------------------------------------------------- reader errors
def test_reader_area_is_required(tmp_path: Path) -> None:
    df = pd.DataFrame({"time/s": [0.0, 1.0], "<I>/mA": [0.1, 0.1]})
    path = _write_export(tmp_path / "x_01_GEIS_C01.txt", df)
    with pytest.raises(TypeError):
        read_biologic_txt(path)  # type: ignore[call-arg]  # area_cm2 omitted


@pytest.mark.parametrize("bad_area", [0.0, -1.0, float("nan"), float("inf")])
def test_reader_rejects_invalid_area(tmp_path: Path, bad_area: float) -> None:
    df = pd.DataFrame({"time/s": [0.0, 1.0], "<I>/mA": [0.1, 0.1]})
    path = _write_export(tmp_path / "x_01_GEIS_C01.txt", df)
    with pytest.raises(ValueError, match="area_cm2"):
        read_biologic_txt(path, area_cm2=bad_area)


def test_reader_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_biologic_txt("does_not_exist_01_GEIS_C01.txt", area_cm2=1.0)


def test_reader_no_header_line(tmp_path: Path) -> None:
    path = tmp_path / "junk.txt"
    path.write_text("no columns here\njust prose\n", encoding="latin1")
    with pytest.raises(ValueError, match="header"):
        read_biologic_txt(path, area_cm2=1.0)


def test_reader_honours_nb_header_lines_marker(tmp_path: Path) -> None:
    path = tmp_path / "marker.txt"
    path.write_text(
        "EC-Lab ASCII FILE\n"
        "Nb header lines : 3\n"
        "time/s\t<I>/mA\n"
        "0.0\t0.5\n"
        "1.0\t0.5\n",
        encoding="latin1",
    )
    loaded = read_biologic_txt(path, area_cm2=2.0)
    assert list(loaded["time/s"]) == [0.0, 1.0]
    assert loaded["cap_mah"].to_numpy()[-1] == pytest.approx(0.5 / 3600.0)


# --------------------------------------------------------- sweep segmentation
def test_sweep_bounds_three_sweeps_last_truncated() -> None:
    full = np.logspace(6, -1, 25)             # 1 MHz -> 0.1 Hz
    cut = full[full >= 100.0]                 # truncated final sweep
    f = np.concatenate([full, full, cut])
    bounds = sweep_bounds(f)
    assert bounds == [(0, 25), (25, 50), (50, 50 + cut.size)]


def test_sweep_bounds_reset_decades_threshold() -> None:
    # The upward step 1 Hz -> 30 Hz is ~1.5 decades: not a reset at the
    # default 2-decade threshold, but a reset at 1 decade.
    f = np.array([100.0, 10.0, 1.0, 30.0, 3.0, 0.3])
    assert sweep_bounds(f) == [(0, 6)]
    assert sweep_bounds(f, reset_decades=1.0) == [(0, 3), (3, 6)]


def test_sweep_bounds_empty_and_invalid() -> None:
    assert sweep_bounds(np.array([])) == []
    with pytest.raises(ValueError, match="positive"):
        sweep_bounds([1000.0, 0.0, 10.0])
    with pytest.raises(ValueError, match="positive"):
        sweep_bounds([1000.0, float("nan")])
    with pytest.raises(ValueError, match="reset_decades"):
        sweep_bounds([1000.0, 10.0], reset_decades=0.0)


def test_segment_sweeps_missing_freq_column() -> None:
    with pytest.raises(ValueError, match="freq"):
        segment_sweeps(pd.DataFrame({"time/s": [0.0, 1.0]}))


def test_segment_sweeps_explicit_cap_col_must_exist() -> None:
    df = pd.DataFrame({"freq/Hz": [1000.0, 100.0]})
    with pytest.raises(ValueError, match="capacity"):
        segment_sweeps(df, cap_col="no_such_column")


# -------------------------------------------- synthetic stream + round trip
def test_geis_stream_is_deterministic() -> None:
    kwargs = dict(n_sweeps=2, points_per_sweep=10, truncate_last_at_hz=50.0)
    a = geis_stream(rng=123, **kwargs)
    b = geis_stream(rng=123, **kwargs)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.sweep_bounds == b.sweep_bounds


def test_segmentation_roundtrip_through_reader(tmp_path: Path) -> None:
    stream = geis_stream(
        n_sweeps=3,
        points_per_sweep=30,
        f_start_hz=1.0e6,
        f_end_hz=0.1,
        truncate_last_at_hz=50.0,
        current_ma=0.25,
        noise_frac=0.01,
        include_broken_q=True,
        include_ewe_modulus=True,
        rng=np.random.default_rng(20260703),
    )
    assert stream.truncated_last
    assert len(stream.sweep_bounds) == 3
    n_full = 30
    assert stream.sweep_bounds[0] == (0, n_full)
    assert stream.sweep_bounds[1] == (n_full, 2 * n_full)
    last_start, last_stop = stream.sweep_bounds[2]
    assert last_start == 2 * n_full and (last_stop - last_start) < n_full

    path = _write_export(
        tmp_path / "synthetic_04_GEIS_C11.txt",
        stream.frame,
        preamble="ASCII EXPORT\nSome instrument metadata line\n",
    )
    loaded = read_biologic_txt(path, area_cm2=0.5)
    assert len(loaded) == len(stream.frame)

    # Column pinning and capacity behaviour on the loaded frame.
    assert resolve_ewe_column(loaded.columns) == COL_EWE
    assert np.all(np.diff(loaded["cap_mah"].to_numpy()) >= 0)
    assert np.diff(loaded["cap_q_mah"].to_numpy()).min() < 0  # broken Q resets

    # Segmentation recovers the ground-truth sweep structure.
    sweeps = segment_sweeps(loaded)
    assert [(s.start, s.stop) for s in sweeps] == stream.sweep_bounds
    assert [s.complete for s in sweeps] == [True, True, False]
    assert [s.n_points for s in sweeps] == [
        b - a for a, b in stream.sweep_bounds
    ]
    assert all(s.starts_at_top is None for s in sweeps)  # no marker given

    # Per-sweep metadata: capacity auto-detected (coulomb-counted areal
    # column preferred), time and capacity means increase sweep to sweep.
    assert all(s.cap_col == "cap_mah_cm2" for s in sweeps)
    times = [s.time_mean_s for s in sweeps]
    caps = [s.cap_mean for s in sweeps]
    assert times == sorted(times) and len(set(times)) == 3
    assert caps == sorted(caps) and len(set(caps)) == 3

    # Explicit start/end markers sharpen the flags without changing bounds.
    marked = segment_sweeps(loaded, f_start_hz=1.0e6, f_end_hz=0.1)
    assert [(s.start, s.stop) for s in marked] == stream.sweep_bounds
    assert [s.complete for s in marked] == [True, True, False]
    assert [s.starts_at_top for s in marked] == [True, True, True]

    # Slicing helper returns per-sweep frames of the right size.
    frames = sweep_frames(loaded, sweeps)
    assert [len(fr) for fr in frames] == [s.n_points for s in sweeps]
