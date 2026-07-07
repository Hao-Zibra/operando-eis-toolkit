# operando-eis-toolkit

Analysis primitives for **operando electrochemical impedance spectroscopy (EIS)**
and **censored reliability** of battery cells: read raw instrument exports, fit
equivalent circuits and distribution-of-relaxation-times (DRT), run
right-censored lifetime statistics, and monitor process stability — all from a
small, dependency-light Python package that ships **no dataset-specific
defaults** (electrode area, rated capacity, and frequency windows are always
explicit parameters).

> **Provenance.** This toolkit packages the analysis methods I designed and
> directed for my operando-EIS reliability program on anode-free solid-state
> cells (2025–2026): censored life-data fitting, EIS sweep segmentation and
> equivalent-circuit fitting, DRT deconvolution, SPC charts, and spec-enforced
> figure export. Implementation was AI-accelerated under my direction; the
> methodological decisions — failure criteria, censoring policy, distribution
> selection, and validation gates — are mine. The package contains methods and
> synthetic data only: no laboratory data and no study results. The underlying
> research program is unpublished (manuscript in preparation); the published
> anchor for the operando method is our ACS Electrochemistry 2026 paper
> (co-first author). — Hao Zheng

## Install

```bash
pip install -e .            # core: numpy, scipy, pandas
pip install -e ".[plot]"    # + matplotlib (figstyle, examples)
pip install -e ".[dev]"     # + pytest, matplotlib (run the tests)
```

Python ≥ 3.10.

## 60-second tour

```bash
python examples/end_to_end_synthetic.py   # all four stages on generated data
python examples/figure_style_demo.py      # spec-compliant PNGs (2200 px wide)
```

`end_to_end_synthetic.py` runs the whole pipeline without any real data:

```
1. EIS   — AICc picks a 2-arc R-CPE fit; reads the model-free valley resistance
2. DRT   — recovers both relaxation-time peaks from the same sweep
3. Reliability — censored Weibull B10/B50 with a profile-likelihood interval
4. SPC + fade  — Nelson-rule scan; fade-model fit with a cycles-to-80 % estimate
```

## What's inside (one entry point per concern)

| Module | Purpose |
|---|---|
| `oeis_toolkit.io.biologic` | Read BioLogic-style tab-separated (latin1) exports. Pins `<Ewe>/V` (never the `\|Ewe\|/V` decoy), coulomb-counts capacity from `\|I\|·dt` when the instrument Q column is unreliable, parses the `_<NN>_<TYPE>_<Cxx>` filename tail. |
| `oeis_toolkit.eis.segmentation` | Split a GEIS/PEIS point stream into frequency sweeps by detecting frequency resets — no hard-coded instrument frequencies. Flags truncated sweeps. |
| `oeis_toolkit.eis.circuit` | Complex NLLS fit of `R_s + (R‖CPE)[+(R‖CPE)]`, AICc 1-vs-2-arc selection, per-parameter uncertainties, and a model-free valley-resistance (IR-correction) proxy over a caller-supplied window. |
| `oeis_toolkit.eis.drt` | Tikhonov-regularized DRT (NNLS, L-curve λ), peak and region resistances. |
| `oeis_toolkit.reliability.censored` | Right-censored MLE for weibull/lognormal/loglogistic/normal/exponential; B-lives with profile-likelihood CIs; Kaplan-Meier; Johnson median ranks; AICc table; optional `lifelines`/`reliability` cross-check. |
| `oeis_toolkit.spc.charts` | Phase-I I-MR and EWMA control charts with Nelson rules 1-4, returned as tidy violation records. |
| `oeis_toolkit.fade` | Per-cycle capacity/CE rollup; linear/exponential/√n fade fit by AICc; cycles-to-fraction projection. |
| `oeis_toolkit.figstyle` | One importable figure spec: 12 pt ticks, uniform line width, zero-anchored quantity axes, and exact-pixel export (`EXPORT_W_PX = TARGET_AI_WIDTH_PX × DPI_EXPORT / 72`). |
| `oeis_toolkit.synthetic` | Deterministic generators (seeded) for every example and test. |

## Design contracts

- **No silent fallbacks.** Ambiguous input raises a typed, actionable error
  (`CircuitFitError`, `InsufficientCyclesError`, …) rather than returning NaN —
  so the pipeline can run unattended and fail loudly instead of quietly wrong.
- **Explicit physics.** No default electrode area or rated capacity anywhere;
  frequency windows are parameters, not constants.
- **Reproducible.** Every stochastic path takes a required seed/generator.
- **Cross-checked statistics.** Custom censored MLE is independently validated
  against `lifelines` and `reliability` when those optional packages are present.

## Tests

```bash
pytest tests/ -q          # 76 tests, synthetic data only, no network, no real data
```

The suite covers filename/column parsing, coulomb counting, sweep segmentation,
circuit + DRT parameter recovery, censored-MLE recovery and CIs against
hand-computed values, Nelson rules and EWMA recursion, and fade-model selection.

## License

MIT — see [LICENSE](LICENSE).
