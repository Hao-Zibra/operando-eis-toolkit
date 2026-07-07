"""operando-eis-toolkit — analysis primitives for operando electrochemical
impedance spectroscopy and censored reliability of battery cells.

A small, dependency-light toolkit assembled from primitives commonly needed
when turning raw galvanostatic/potentiostatic EIS exports into mechanism and
reliability conclusions:

- ``io``          — read BioLogic-style tab-separated exports; coulomb-counted
                    capacity; filename-tail parsing.
- ``eis``         — sweep segmentation, R-CPE equivalent-circuit fitting with
                    AICc model selection, Tikhonov DRT.
- ``reliability`` — right-censored MLE across five lifetime distributions,
                    B-lives with profile-likelihood CIs, Kaplan-Meier.
- ``spc``         — I-MR / EWMA control charts with Nelson rules.
- ``fade``        — per-cycle capacity/CE rollup and fade-model fitting.
- ``figstyle``    — one importable figure spec (sizes, line widths, exact
                    export width) for publication/industrial plots.
- ``synthetic``   — deterministic generators for examples and tests.

Every entry point requires physical parameters (electrode area, rated capacity,
frequency windows) explicitly — the toolkit ships no dataset-specific defaults.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
