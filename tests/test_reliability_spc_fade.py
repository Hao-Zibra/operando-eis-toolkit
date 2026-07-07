"""Tests for oeis_toolkit.reliability.censored, oeis_toolkit.spc.charts and
oeis_toolkit.fade.

All data here is synthetic, generated inside the tests from fixed seeds.
Hand-computed reference values are derived in comments next to each test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oeis_toolkit.fade import (  # noqa: E402
    FadeError,
    InsufficientCyclesError,
    fit_fade,
)
from oeis_toolkit.reliability.censored import (  # noqa: E402
    InvalidDataError,
    aicc_table,
    fit_censored,
    kaplan_meier,
    median_ranks,
    profile_ci,
    profile_ci_b_life,
    quantile,
)
from oeis_toolkit.spc.charts import (  # noqa: E402
    SPCError,
    ewma,
    imr_limits,
    nelson_rules,
    violations_frame,
)

# --------------------------------------------------------------------------- #
# synthetic censored-Weibull sample (generated here; no external data)
# --------------------------------------------------------------------------- #

TRUE_BETA = 2.0
TRUE_ETA = 10.0
CENSOR_TIME = 10.5  # S(10.5) = exp(-(1.05)^2) ~ 0.33 -> roughly one third censored


def _censored_weibull_sample(
    n: int, seed: int, beta: float = TRUE_BETA, eta: float = TRUE_ETA
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lifetimes = eta * rng.weibull(beta, size=n)
    event = (lifetimes <= CENSOR_TIME).astype(int)
    x = np.minimum(lifetimes, CENSOR_TIME)
    return x, event


# --------------------------------------------------------------------------- #
# censored MLE
# --------------------------------------------------------------------------- #


class TestCensoredWeibullMLE:
    def test_parameter_recovery_under_censoring(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        frac_censored = 1.0 - event.mean()
        assert 0.25 <= frac_censored <= 0.45  # the intended ~30-40% regime

        fit = fit_censored(x, event, "weibull")
        beta_hat = fit.params["beta"]
        eta_hat = fit.params["eta"]
        assert abs(beta_hat - TRUE_BETA) / TRUE_BETA < 0.20
        assert abs(eta_hat - TRUE_ETA) / TRUE_ETA < 0.10
        assert fit.n_total == 250
        assert fit.n_failures == int(event.sum())
        assert fit.n_censored == 250 - int(event.sum())

    def test_b_life_matches_closed_form(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        fit = fit_censored(x, event, "weibull")
        beta_hat, eta_hat = fit.params["beta"], fit.params["eta"]
        expected_b10 = eta_hat * (-np.log(0.9)) ** (1.0 / beta_hat)
        assert fit.b_life(0.10) == pytest.approx(expected_b10, rel=1e-12)

    def test_rejects_all_censored_sample(self) -> None:
        with pytest.raises(InvalidDataError):
            fit_censored([1.0, 2.0, 3.0], [0, 0, 0], "weibull")

    def test_rejects_bad_event_codes(self) -> None:
        with pytest.raises(InvalidDataError):
            fit_censored([1.0, 2.0, 3.0], [1, 2, 1], "weibull")


class TestProfileCI:
    def test_beta_ci_brackets_truth(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        ci = profile_ci(x, event, parameter="beta", level=0.99)
        assert np.isfinite(ci.lower) and np.isfinite(ci.upper)
        assert ci.lower < ci.estimate < ci.upper
        assert ci.lower < TRUE_BETA < ci.upper

    def test_eta_ci_brackets_truth(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        ci = profile_ci(x, event, parameter="eta", level=0.99)
        assert ci.lower < ci.estimate < ci.upper
        assert ci.lower < TRUE_ETA < ci.upper

    def test_b10_ci_brackets_truth(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        true_b10 = TRUE_ETA * (-np.log(0.9)) ** (1.0 / TRUE_BETA)
        ci = profile_ci_b_life(x, event, p=0.10, level=0.99)
        assert ci.parameter == "B10"
        assert ci.lower < ci.estimate < ci.upper
        assert ci.lower < true_b10 < ci.upper

    def test_estimate_matches_mle_quantile(self) -> None:
        x, event = _censored_weibull_sample(n=250, seed=20260703)
        fit = fit_censored(x, event, "weibull")
        ci = profile_ci_b_life(x, event, p=0.10, level=0.95)
        assert ci.estimate == pytest.approx(fit.b_life(0.10), rel=1e-6)


class TestKaplanMeier:
    def test_hand_computed_six_point_example(self) -> None:
        # x = [1,2,3,4,5,6], event = [1,1,0,1,0,1]
        # t=1: 6 at risk, 1 death -> S = 5/6
        # t=2: 5 at risk, 1 death -> S = 5/6 * 4/5 = 2/3
        # t=3: censored only -> no step
        # t=4: 3 at risk, 1 death -> S = 2/3 * 2/3 = 4/9
        # t=5: censored only -> no step
        # t=6: 1 at risk, 1 death -> S = 0
        km = kaplan_meier([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 0, 1])
        np.testing.assert_allclose(km.times, [0.0, 1.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(
            km.survival, [1.0, 5.0 / 6.0, 2.0 / 3.0, 4.0 / 9.0, 0.0]
        )

    def test_survival_at_censored_times(self) -> None:
        km = kaplan_meier([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 0, 1])
        # step function is flat through censored-only times
        assert km.survival_at(3.0) == pytest.approx(2.0 / 3.0)
        assert km.survival_at(5.0) == pytest.approx(4.0 / 9.0)
        np.testing.assert_allclose(
            km.survival_at([0.5, 3.0, 4.5]), [1.0, 2.0 / 3.0, 4.0 / 9.0]
        )


class TestMedianRanks:
    def test_hand_checked_johnson_table(self) -> None:
        # n = 4, x = [10, 20, 30, 40], event = [1, 0, 1, 1]
        # failure at 10: increment = (4+1-0)/(1+(4-0)) = 1        -> adj = 1
        #   F = (1 - 0.3) / 4.4          = 0.1590909...
        # censored at 20: no point, but the next increment grows
        # failure at 30: increment = (5-1)/(1+(4-2)) = 4/3        -> adj = 7/3
        #   F = (7/3 - 0.3) / 4.4        = 0.4621212...
        # failure at 40: increment = (5-7/3)/(1+(4-3)) = 4/3      -> adj = 11/3
        #   F = (11/3 - 0.3) / 4.4       = 0.7651515...
        times, ranks = median_ranks([10, 20, 30, 40], [1, 0, 1, 1])
        np.testing.assert_allclose(times, [10.0, 30.0, 40.0])
        np.testing.assert_allclose(
            ranks,
            [
                (1.0 - 0.3) / 4.4,
                (7.0 / 3.0 - 0.3) / 4.4,
                (11.0 / 3.0 - 0.3) / 4.4,
            ],
        )

    def test_uncensored_ranks_reduce_to_bernard(self) -> None:
        # with no censoring the adjusted ranks are 1..n
        n = 5
        _, ranks = median_ranks(np.arange(1.0, n + 1.0), np.ones(n, dtype=int))
        expected = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        np.testing.assert_allclose(ranks, expected)


class TestAICcComparison:
    def test_aicc_rejects_exponential_for_wearout_data(self) -> None:
        # steep wear-out (shape 3) is grossly mis-described by the
        # constant-hazard exponential
        rng = np.random.default_rng(11)
        lifetimes = 10.0 * rng.weibull(3.0, size=120)
        censor = 11.0
        event = (lifetimes <= censor).astype(int)
        x = np.minimum(lifetimes, censor)

        table = aicc_table(x, event)
        assert table["aicc"].is_monotonic_increasing
        assert table.iloc[0]["distribution"] != "exponential"

        by_dist = table.set_index("distribution")
        assert by_dist.loc["weibull", "aicc"] < by_dist.loc["exponential", "aicc"]
        assert by_dist.loc["exponential", "delta_aicc"] > 10.0

    def test_table_reports_b_lives(self) -> None:
        x, event = _censored_weibull_sample(n=120, seed=3)
        table = aicc_table(x, event, b_lives=(0.10, 0.50))
        assert {"b10", "b50"} <= set(table.columns)
        wei = table.set_index("distribution").loc["weibull"]
        expected = quantile(
            "weibull", {"beta": wei["shape"], "eta": wei["scale"]}, 0.10
        )
        assert wei["b10"] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# SPC: I-MR, EWMA, Nelson rules
# --------------------------------------------------------------------------- #


class TestIMRLimits:
    def test_hand_computed_limits(self) -> None:
        # x = [1..5]: MR = [1,1,1,1], MRbar = 1, CL = 3
        lim = imr_limits([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lim.center == pytest.approx(3.0)
        assert lim.ucl == pytest.approx(3.0 + 2.66)
        assert lim.lcl == pytest.approx(3.0 - 2.66)
        assert lim.mr_center == pytest.approx(1.0)
        assert lim.mr_ucl == pytest.approx(3.267)
        assert lim.sigma == pytest.approx(1.0 / 1.128)
        assert lim.n_points == 5

    def test_constant_series_rejected(self) -> None:
        with pytest.raises(SPCError):
            imr_limits([2.0, 2.0, 2.0, 2.0])


class TestNelsonRules:
    def test_rule1_fires_on_outlier(self) -> None:
        x = np.zeros(20)
        x[7] = 4.0
        hits = nelson_rules(x, center=0.0, sigma=1.0)
        assert len(hits) == 1
        v = hits[0]
        assert (v.rule, v.index, v.value) == (1, 7, 4.0)
        assert "3 sigma" in v.rule_text

    def test_rule2_fires_on_nine_same_side(self) -> None:
        x = np.full(12, 0.5)
        hits = nelson_rules(x, center=0.0, sigma=1.0)
        rule2 = [v.index for v in hits if v.rule == 2]
        # the run reaches 9 at index 8 and every extension is also flagged
        assert rule2 == [8, 9, 10, 11]
        assert all(v.rule == 2 for v in hits)

    def test_rule3_fires_on_six_trending(self) -> None:
        x = np.arange(8, dtype=float)
        hits = nelson_rules(x, center=3.5, sigma=10.0)
        rule3 = [v.index for v in hits if v.rule == 3]
        assert rule3 == [5, 6, 7]
        assert all(v.rule == 3 for v in hits)

    def test_rule4_fires_on_fourteen_alternating(self) -> None:
        x = np.array([0.0, 1.0] * 7)  # 14 points, strictly alternating
        hits = nelson_rules(x, center=0.5, sigma=10.0)
        assert [(v.rule, v.index) for v in hits] == [(4, 13)]

    def test_silent_on_in_control_series(self) -> None:
        # deterministic in-control pattern: short runs on each side, no
        # long trends, no long alternation, everything well inside 3 sigma;
        # the small noise cannot flip any sign or side
        rng = np.random.default_rng(99)
        base = np.tile([0.5, 1.0, -0.5, -1.0, 0.8, -0.3], 4)
        x = base + 0.05 * rng.standard_normal(base.size)
        hits = nelson_rules(x, center=0.0, sigma=1.0)
        assert hits == []

    def test_violations_frame_is_tidy(self) -> None:
        x = np.zeros(10)
        x[4] = -5.0
        frame = violations_frame(nelson_rules(x, center=0.0, sigma=1.0))
        assert list(frame.columns) == ["index", "rule", "value", "rule_text"]
        assert frame.loc[0, "index"] == 4
        assert frame.loc[0, "rule"] == 1


class TestEWMA:
    def test_recursion_matches_hand_rolled_loop(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(5.0, 1.0, size=40)
        lam, big_l, target, sigma = 0.3, 2.5, 5.0, 1.0
        chart = ewma(x, lam=lam, L=big_l, target=target, sigma=sigma)

        z_expected = np.empty_like(x)
        prev = target
        for i, xi in enumerate(x):
            prev = lam * xi + (1.0 - lam) * prev
            z_expected[i] = prev
        np.testing.assert_allclose(chart.z, z_expected, rtol=0, atol=1e-12)

        idx = np.arange(1, len(x) + 1, dtype=float)
        half = big_l * sigma * np.sqrt(
            lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2.0 * idx))
        )
        np.testing.assert_allclose(chart.ucl, target + half, atol=1e-12)
        np.testing.assert_allclose(chart.lcl, target - half, atol=1e-12)

    def test_violation_indices_match_manual_check(self) -> None:
        x = np.zeros(15)
        x[10:] = 3.0  # sustained shift the EWMA must catch
        chart = ewma(x, lam=0.4, L=3.0, target=0.0, sigma=0.5)
        manual = [
            i
            for i in range(len(x))
            if chart.z[i] > chart.ucl[i] or chart.z[i] < chart.lcl[i]
        ]
        assert chart.violations() == manual
        assert manual  # the shift is definitely flagged

    def test_invalid_lambda_rejected(self) -> None:
        with pytest.raises(SPCError):
            ewma([1.0, 2.0, 3.0], lam=0.0)


# --------------------------------------------------------------------------- #
# fade fitting and projection
# --------------------------------------------------------------------------- #


class TestFadeFitting:
    def test_recovers_linear_truth_and_projection(self) -> None:
        rng = np.random.default_rng(5)
        cycles = np.arange(50, dtype=float)
        c0_true, k_true = 2.0, 0.005
        capacity = c0_true - k_true * cycles + rng.normal(0.0, 0.001, cycles.size)

        fit = fit_fade(cycles, capacity)
        assert fit.best_model == "linear"
        c0_hat, k_hat = fit.best.params
        assert c0_hat == pytest.approx(c0_true, rel=0.01)
        assert k_hat == pytest.approx(k_true, rel=0.10)

        proj = fit.cycles_to_fraction(0.80)
        # analytic truth: c0 (1 - 0.8) / k = 0.4 / 0.005 = 80 cycles
        assert proj.cycles == pytest.approx(80.0, abs=5.0)
        assert proj.extrapolated is False  # 80 < 2 * 49 observed window

    def test_recovers_exponential_truth_and_projection(self) -> None:
        rng = np.random.default_rng(8)
        cycles = np.arange(60, dtype=float)
        c0_true, n0_true = 2.0, 300.0
        capacity = c0_true * np.exp(-cycles / n0_true) + rng.normal(
            0.0, 0.001, cycles.size
        )

        fit = fit_fade(cycles, capacity)
        assert fit.best_model == "exponential"
        c0_hat, n0_hat = fit.best.params
        assert c0_hat == pytest.approx(c0_true, rel=0.01)
        assert n0_hat == pytest.approx(n0_true, rel=0.15)

        proj = fit.cycles_to_fraction(0.80)
        # analytic truth: -n0 ln(0.8) = 300 * 0.22314... ~ 66.9 cycles
        expected = -n0_true * np.log(0.8)
        assert proj.cycles == pytest.approx(expected, abs=5.0)
        assert proj.extrapolated is False  # 66.9 < 2 * 59 observed window

    def test_refuses_too_few_cycles_with_typed_exception(self) -> None:
        with pytest.raises(InsufficientCyclesError):
            fit_fade([0.0, 1.0, 2.0], [2.0, 1.99, 1.98])

    def test_nan_cycles_count_against_minimum(self) -> None:
        cycles = [0.0, 1.0, 2.0, np.nan, np.nan]
        capacity = [2.0, 1.99, 1.98, 1.97, 1.96]
        with pytest.raises(InsufficientCyclesError):
            fit_fade(cycles, capacity)

    def test_invalid_fraction_rejected(self) -> None:
        rng = np.random.default_rng(2)
        cycles = np.arange(10, dtype=float)
        capacity = 2.0 - 0.01 * cycles + rng.normal(0.0, 0.001, cycles.size)
        fit = fit_fade(cycles, capacity)
        with pytest.raises(FadeError):
            fit.cycles_to_fraction(1.5)
