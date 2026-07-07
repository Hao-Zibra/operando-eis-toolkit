"""Synthetic-data tests for oeis_toolkit.eis.circuit and oeis_toolkit.eis.drt.

All spectra are generated locally from known R_s / (R, tau, alpha) arc
parameters -- no measured data, no external synthetic module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the src/ layout importable without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oeis_toolkit.eis.circuit import (  # noqa: E402
    CircuitFitError,
    circuit_impedance,
    fit_best_model,
    fit_circuit,
    valley_resistance,
)
from oeis_toolkit.eis.drt import (  # noqa: E402
    DRTError,
    compute_drt,
    find_drt_peaks,
    region_resistance,
)

# Instrument-style descending sweeps.
FREQ_DENSE = np.logspace(6, -2, 81)  # 1 MHz .. 10 mHz, 10 pts/decade
FREQ_DRT = np.logspace(6, -3, 91)  # 1 MHz .. 1 mHz, 10 pts/decade


def _q_from_tau(resistance: float, tau_s: float, alpha: float) -> float:
    """CPE coefficient giving relaxation time tau: tau = (R*Q)**(1/alpha)."""
    return tau_s**alpha / resistance


def _synthetic_spectrum(
    freq_hz: np.ndarray,
    r_series: float,
    arcs: list[tuple[float, float, float]],
    noise_frac: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(Re Z, -Im Z) of R_s + sum of R||CPE arcs given as (R, tau_s, alpha),
    with optional proportional Gaussian noise."""
    z = circuit_impedance(
        freq_hz,
        r_series,
        [(r, _q_from_tau(r, tau, a), a) for r, tau, a in arcs],
    )
    if noise_frac > 0.0:
        rng = np.random.default_rng(seed)
        scale = noise_frac * np.abs(z)
        z = z + scale * rng.standard_normal(z.size)
        z = z + 1j * scale * rng.standard_normal(z.size)
    return z.real, -z.imag


# --------------------------------------------------------------------------
# Circuit fitting
# --------------------------------------------------------------------------
class TestCircuitFit:
    def test_two_arc_fit_recovers_true_parameters(self):
        true_arcs = [(50.0, 2e-5, 0.90), (200.0, 0.2, 0.85)]
        re_, nim = _synthetic_spectrum(
            FREQ_DENSE, 10.0, true_arcs, noise_frac=0.002, seed=1
        )
        res = fit_circuit(FREQ_DENSE, re_, nim, n_arcs=2)

        assert res.n_arcs == 2
        assert res.n_points_used == FREQ_DENSE.size
        assert res.r_series_ohm == pytest.approx(10.0, rel=0.05)

        # Result orders arcs by descending characteristic frequency,
        # i.e. ascending relaxation time: fast arc first.
        fast, slow = res.arcs
        assert fast.time_constant_s < slow.time_constant_s
        assert fast.resistance_ohm == pytest.approx(50.0, rel=0.05)
        assert slow.resistance_ohm == pytest.approx(200.0, rel=0.05)
        assert fast.time_constant_s == pytest.approx(2e-5, rel=0.3)
        assert slow.time_constant_s == pytest.approx(0.2, rel=0.3)
        assert abs(fast.alpha - 0.90) < 0.05
        assert abs(slow.alpha - 0.85) < 0.05

        # r_total and misfit sanity.
        assert res.r_total_ohm == pytest.approx(260.0, rel=0.05)
        assert res.rmse_pct < 2.0

    def test_uncertainties_reported_for_well_conditioned_fit(self):
        true_arcs = [(50.0, 2e-5, 0.90), (200.0, 0.2, 0.85)]
        re_, nim = _synthetic_spectrum(
            FREQ_DENSE, 10.0, true_arcs, noise_frac=0.002, seed=1
        )
        res = fit_circuit(FREQ_DENSE, re_, nim, n_arcs=2)
        assert res.r_series_std_ohm is not None
        assert 0.0 < res.r_series_std_ohm < 0.1 * res.r_series_ohm
        for arc in res.arcs:
            if arc.resistance_std_ohm is not None:
                assert arc.resistance_std_ohm > 0.0

    def test_fitted_model_reproduces_spectrum(self):
        true_arcs = [(50.0, 2e-5, 0.90), (200.0, 0.2, 0.85)]
        re_, nim = _synthetic_spectrum(FREQ_DENSE, 10.0, true_arcs)
        res = fit_circuit(FREQ_DENSE, re_, nim, n_arcs=2)
        z_fit = res.impedance(FREQ_DENSE)
        z_true = re_ - 1j * nim
        rel = np.abs(z_fit - z_true) / np.abs(z_true)
        assert float(np.max(rel)) < 0.02

    def test_aicc_selects_two_arcs_for_two_arc_truth(self):
        true_arcs = [(50.0, 2e-5, 0.90), (200.0, 0.2, 0.85)]
        re_, nim = _synthetic_spectrum(
            FREQ_DENSE, 10.0, true_arcs, noise_frac=0.005, seed=2
        )
        res = fit_best_model(FREQ_DENSE, re_, nim)
        assert res.n_arcs == 2
        assert 2 in res.aicc_by_model
        if 1 in res.aicc_by_model:
            assert res.aicc_by_model[2] < res.aicc_by_model[1]

    def test_aicc_selects_one_arc_for_one_arc_truth(self):
        # Sparse sampling keeps the AICc small-sample penalty decisive
        # against noise-fitting by the superfluous second arc.
        freq = np.logspace(5, -1, 17)
        re_, nim = _synthetic_spectrum(
            freq, 20.0, [(100.0, 1e-3, 0.90)], noise_frac=0.005, seed=3
        )
        res = fit_best_model(freq, re_, nim)
        assert res.n_arcs == 1
        assert res.r_series_ohm == pytest.approx(20.0, rel=0.1)
        assert res.arcs[0].resistance_ohm == pytest.approx(100.0, rel=0.1)

    def test_too_few_points_raises_typed_error_with_diagnostics(self):
        freq = np.logspace(4, 0, 5)
        re_, nim = _synthetic_spectrum(freq, 10.0, [(50.0, 1e-3, 0.9)])
        with pytest.raises(CircuitFitError) as excinfo:
            fit_circuit(freq, re_, nim, n_arcs=1)
        assert excinfo.value.diagnostics["n_points"] == 5
        assert "n_required" in excinfo.value.diagnostics

    def test_all_candidates_failing_raises(self):
        freq = np.logspace(4, 0, 5)
        re_, nim = _synthetic_spectrum(freq, 10.0, [(50.0, 1e-3, 0.9)])
        with pytest.raises(CircuitFitError) as excinfo:
            fit_best_model(freq, re_, nim)
        assert "model_errors" in excinfo.value.diagnostics

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError):
            fit_circuit([1.0, 10.0, 100.0], [1.0, 2.0], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------
# Valley resistance (IR-correction proxy)
# --------------------------------------------------------------------------
class TestValleyResistance:
    def test_valley_approximates_rs_plus_r1_for_separated_arcs(self):
        # Arcs ~5 decades apart in characteristic frequency (~10 kHz vs
        # ~0.1 Hz); the inter-arc valley Re(Z) should sit near R_s + R1.
        r_s, r1, r2 = 10.0, 50.0, 200.0
        true_arcs = [(r1, 1.6e-5, 0.95), (r2, 1.6, 0.95)]
        re_, nim = _synthetic_spectrum(
            FREQ_DENSE, r_s, true_arcs, noise_frac=0.001, seed=4
        )
        v = valley_resistance(
            FREQ_DENSE, re_, nim, f_min_hz=1.0, f_max_hz=1e3
        )
        assert v.resistance_ohm == pytest.approx(r_s + r1, rel=0.10)
        assert 1.0 <= v.frequency_hz <= 1e3
        assert v.n_points_in_window > 0

    def test_window_outside_data_raises_typed_error(self):
        re_, nim = _synthetic_spectrum(FREQ_DENSE, 10.0, [(50.0, 1e-3, 0.9)])
        with pytest.raises(CircuitFitError):
            valley_resistance(
                FREQ_DENSE, re_, nim, f_min_hz=1e7, f_max_hz=1e8
            )

    def test_invalid_window_raises_value_error(self):
        re_, nim = _synthetic_spectrum(FREQ_DENSE, 10.0, [(50.0, 1e-3, 0.9)])
        with pytest.raises(ValueError):
            valley_resistance(
                FREQ_DENSE, re_, nim, f_min_hz=100.0, f_max_hz=1.0
            )


# --------------------------------------------------------------------------
# DRT
# --------------------------------------------------------------------------
TRUE_TAU_FAST = 1e-4
TRUE_TAU_SLOW = 1.0
TRUE_R_INF = 5.0
TRUE_R_FAST = 40.0
TRUE_R_SLOW = 100.0


@pytest.fixture(scope="module")
def drt_result():
    true_arcs = [
        (TRUE_R_FAST, TRUE_TAU_FAST, 0.92),
        (TRUE_R_SLOW, TRUE_TAU_SLOW, 0.90),
    ]
    re_, nim = _synthetic_spectrum(
        FREQ_DRT, TRUE_R_INF, true_arcs, noise_frac=0.002, seed=5
    )
    return compute_drt(FREQ_DRT, re_, nim)


class TestDRT:
    def test_lcurve_lambda_is_finite_and_positive(self, drt_result):
        assert np.isfinite(drt_result.lam)
        assert drt_result.lam > 0
        assert drt_result.lambda_grid is not None
        assert drt_result.lam in drt_result.lambda_grid
        assert (
            drt_result.lcurve_log_residual.shape
            == drt_result.lambda_grid.shape
        )
        assert (
            drt_result.lcurve_log_solution_norm.shape
            == drt_result.lambda_grid.shape
        )

    def test_peak_positions_recover_true_time_constants(self, drt_result):
        peaks = find_drt_peaks(
            drt_result.tau_s, drt_result.gamma_ohm, min_fraction=0.05
        )
        assert len(peaks) >= 2
        # Two dominant peaks, in ascending relaxation-time order.
        top2 = sorted(
            peaks, key=lambda p: p.resistance_ohm, reverse=True
        )[:2]
        fast, slow = sorted(top2, key=lambda p: p.tau_s)
        assert TRUE_TAU_FAST / 3 <= fast.tau_s <= TRUE_TAU_FAST * 3
        assert TRUE_TAU_SLOW / 3 <= slow.tau_s <= TRUE_TAU_SLOW * 3

    def test_resistances_recovered(self, drt_result):
        assert drt_result.r_pol_ohm == pytest.approx(
            TRUE_R_FAST + TRUE_R_SLOW, rel=0.25
        )
        assert drt_result.r_inf_ohm == pytest.approx(TRUE_R_INF, rel=0.3)
        assert drt_result.residual_pct < 5.0

    def test_region_resistance_attributes_by_time_window(self, drt_result):
        r_fast = region_resistance(
            drt_result.tau_s,
            drt_result.gamma_ohm,
            tau_min_s=TRUE_TAU_FAST / 10 ** 1.5,
            tau_max_s=TRUE_TAU_FAST * 10 ** 1.5,
        )
        r_slow = region_resistance(
            drt_result.tau_s,
            drt_result.gamma_ohm,
            tau_min_s=TRUE_TAU_SLOW / 10 ** 1.5,
            tau_max_s=TRUE_TAU_SLOW * 10 ** 1.5,
        )
        assert r_slow > r_fast
        assert 0.5 * TRUE_R_FAST <= r_fast <= 1.5 * TRUE_R_FAST
        assert 0.5 * TRUE_R_SLOW <= r_slow <= 1.5 * TRUE_R_SLOW

    def test_fixed_lambda_skips_lcurve(self):
        true_arcs = [(TRUE_R_FAST, TRUE_TAU_FAST, 0.92)]
        re_, nim = _synthetic_spectrum(
            FREQ_DRT, TRUE_R_INF, true_arcs, noise_frac=0.002, seed=6
        )
        res = compute_drt(FREQ_DRT, re_, nim, lam=1e-2)
        assert res.lam == pytest.approx(1e-2)
        assert res.lambda_grid is None
        assert res.lcurve_log_residual is None

    def test_too_few_points_raises_typed_error(self):
        freq = np.logspace(3, 1, 6)
        re_, nim = _synthetic_spectrum(freq, 5.0, [(40.0, 1e-3, 0.9)])
        with pytest.raises(DRTError) as excinfo:
            compute_drt(freq, re_, nim)
        assert excinfo.value.diagnostics["n_points"] == 6

    def test_region_window_outside_grid_raises(self, drt_result):
        with pytest.raises(ValueError):
            region_resistance(
                drt_result.tau_s,
                drt_result.gamma_ohm,
                tau_min_s=1e12,
                tau_max_s=1e13,
            )
