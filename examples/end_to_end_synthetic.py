#!/usr/bin/env python3
"""End-to-end demo on synthetic data — no real dataset required.

Runs the toolkit's four analysis stages against generated data so a reader can
see the full workflow and confirm the install works:

    1. EIS      — fit an R-CPE circuit to a synthetic Nyquist sweep and read the
                  model-free valley resistance.
    2. DRT      — recover relaxation-time peaks from the same sweep.
    3. Reliability — right-censored Weibull MLE on synthetic lifetimes; B10/B50
                  with a profile-likelihood interval; AICc across distributions.
    4. SPC + fade — Nelson-rule scan on a control series; fade-model fit with a
                  cycles-to-80 % projection.

Run:  python examples/end_to_end_synthetic.py
"""

from __future__ import annotations

import numpy as np

from oeis_toolkit import synthetic, fade
from oeis_toolkit.eis import circuit, drt
from oeis_toolkit.reliability import censored
from oeis_toolkit.spc import charts

SEED = 20240703  # fixed so the demo is reproducible
FREQ_HZ = np.logspace(6, -1, 70)  # ~1 MHz -> 0.1 Hz, generic sweep grid


def demo_eis() -> None:
    print("== 1. EIS equivalent-circuit fit ==")
    ds = synthetic.nyquist_dataset(
        FREQ_HZ,
        r_s=45.0,
        arcs=[(120.0, 1.0e-6, 0.88), (600.0, 3.0e-4, 0.80)],
        noise_frac=0.01,
        rng=SEED,
    )
    f, re, neg_im = ds["freq/Hz"], ds["Re(Z)/Ohm"], ds["-Im(Z)/Ohm"]

    best = circuit.fit_best_model(f, re, neg_im, candidate_n_arcs=(1, 2))
    print(f"   AICc selected: {best.n_arcs}-arc model  "
          f"(R_s = {best.r_series_ohm:.1f} Ohm, RMSE = {best.rmse_pct:.2f} %)")
    for i, arc in enumerate(best.arcs, 1):
        print(f"   arc {i}: R = {arc.resistance_ohm:7.1f} Ohm, "
              f"tau = {arc.time_constant_s:.2e} s")
    valley = circuit.valley_resistance(f, re, neg_im, f_min_hz=1.0, f_max_hz=5.0e3)
    print(f"   valley R (1-5000 Hz, IR proxy) = {valley.resistance_ohm:.1f} Ohm")


def demo_drt() -> None:
    print("\n== 2. DRT relaxation-time peaks ==")
    ds = synthetic.nyquist_dataset(
        FREQ_HZ,
        r_s=45.0,
        arcs=[(120.0, 1.0e-6, 0.90), (600.0, 3.0e-4, 0.90)],
        noise_frac=0.005,
        rng=SEED,
    )
    result = drt.compute_drt(ds["freq/Hz"], ds["Re(Z)/Ohm"], ds["-Im(Z)/Ohm"])
    peaks = drt.find_drt_peaks(result.tau_s, result.gamma_ohm)
    print(f"   lambda (L-curve) = {result.lam:.2e}; "
          f"R_pol = {result.r_pol_ohm:.0f} Ohm; {len(peaks)} peak(s):")
    for pk in peaks:
        print(f"   tau = {pk.tau_s:.2e} s, R = {pk.resistance_ohm:6.1f} Ohm "
              f"({pk.fraction * 100:.0f} % of R_pol)")


def demo_reliability() -> None:
    print("\n== 3. Censored-Weibull reliability ==")
    sample = synthetic.weibull_censored_sample(
        beta=2.5, eta=1.5, n=40, censor_time=1.6, rng=SEED
    )
    x, event = sample["time"], sample["event"]
    fit = censored.fit_censored(x, event, distribution="weibull")
    b10 = censored.quantile("weibull", fit.params, 0.10)
    b50 = censored.quantile("weibull", fit.params, 0.50)
    ci = censored.profile_ci_b_life(x, event, p=0.10, level=0.90)
    print(f"   n = {fit.n_total} ({fit.n_failures} failures / "
          f"{fit.n_total - fit.n_failures} censored)")
    print(f"   beta = {fit.params['beta']:.2f}, eta = {fit.params['eta']:.3f}")
    print(f"   B10 = {b10:.3f}  [{ci.lower:.3f}, {ci.upper:.3f}] (90% profile)")
    print(f"   B50 = {b50:.3f}")
    table = censored.aicc_table(x, event)
    ranked = ", ".join(
        f"{row.distribution}({row.aicc:.1f})"
        for row in table.sort_values("aicc").head(3).itertuples()
    )
    print(f"   AICc best three: {ranked}")


def demo_spc_fade() -> None:
    print("\n== 4. SPC + fade ==")
    series = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9,
                       10.6, 10.7, 10.8, 10.9, 11.0, 11.1])  # drift at the tail
    limits = charts.imr_limits(series)  # Phase-I center + sigma from I-MR
    violations = charts.nelson_rules(series, center=limits.center, sigma=limits.sigma)
    rules_hit = sorted({v.rule for v in violations})
    print(f"   I-MR center = {limits.center:.2f}, UCL = {limits.ucl:.2f}")
    print(f"   Nelson rules triggered: {rules_hit or 'none'}")

    fade_series = synthetic.capacity_fade_series(
        n_cycles=30, q0=2.0, fade_rate=0.02, model="exp",
        noise_sd=0.01, rng=SEED,
    )
    result = fade.fit_fade(fade_series["cycle"], fade_series["capacity"])
    proj = result.cycles_to_fraction(0.80)
    print(f"   fade model (AICc): {result.best_model}")
    print(f"   observed fade rate: "
          f"{result.observed_fade_rate_pct_per_cycle:.2f} %/cycle")
    print(f"   cycles to 80%: {proj.cycles:.0f}"
          f"{' (extrapolated)' if proj.extrapolated else ''}")


def main() -> None:
    demo_eis()
    demo_drt()
    demo_reliability()
    demo_spc_fade()
    print("\nAll four stages ran on synthetic data.")


if __name__ == "__main__":
    main()
