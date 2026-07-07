#!/usr/bin/env python3
"""Figure-style kit demo — one spec, two spec-compliant PNGs on synthetic data.

Shows the :mod:`oeis_toolkit.figstyle` conventions in action: 12 pt ticks,
uniform line width, capacity axis anchored at zero, and a PNG written exactly
``EXPORT_W_PX = TARGET_AI_WIDTH_PX * DPI_EXPORT / 72`` pixels wide. Writes into
an ``example_figures/`` directory next to this script.

Run:  python examples/figure_style_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oeis_toolkit import synthetic
from oeis_toolkit.figstyle import FigureSpec, apply_style, save_figure, zero_floor

OUTDIR = Path(__file__).resolve().parent / "example_figures"


def main() -> None:
    spec = FigureSpec()  # 12 pt ticks, lw 1.8, 300 dpi, 528 pt -> 2200 px
    apply_style(spec)
    import matplotlib.pyplot as plt

    print(f"spec export width = {spec.export_width_px()} px "
          f"(= {spec.target_ai_width_px} * {spec.dpi_export} / 72)")

    # --- (1) Nyquist plot -----------------------------------------------
    ds = synthetic.nyquist_dataset(
        np.logspace(6, -1, 70), r_s=45.0,
        arcs=[(120.0, 1e-6, 0.88), (600.0, 3e-4, 0.80)],
        noise_frac=0.01, rng=7,
    )
    fig, ax = plt.subplots(figsize=spec.figsize_in())
    ax.plot(ds["Re(Z)/Ohm"], ds["-Im(Z)/Ohm"], marker="o", markersize=3)
    ax.set_xlabel("Re(Z) / Ohm")
    ax.set_ylabel("-Im(Z) / Ohm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Synthetic Nyquist (2-arc R-CPE)")
    nyq = save_figure(fig, OUTDIR / "demo_nyquist.png", spec)
    plt.close(fig)

    # --- (2) capacity-fade plot with zero-anchored y --------------------
    fade = synthetic.capacity_fade_series(
        n_cycles=40, q0=2.0, fade_rate=0.015, model="exp", noise_sd=0.01, rng=7
    )
    fig, ax = plt.subplots(figsize=spec.figsize_in())
    ax.plot(fade["cycle"], fade["capacity"], marker="s", markersize=3)
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity / mAh cm$^{-2}$")
    ax.set_title("Synthetic capacity fade (axis anchored at 0)")
    zero_floor(ax, axis="y")
    fade_png = save_figure(fig, OUTDIR / "demo_fade.png", spec)
    plt.close(fig)

    # --- confirm the export-width contract on disk ----------------------
    try:
        from PIL import Image
        for p in (nyq, fade_png):
            with Image.open(p) as im:
                assert im.width == spec.export_width_px(), (p, im.width)
            print(f"wrote {p.name}: {spec.export_width_px()} px wide (verified)")
    except ImportError:
        for p in (nyq, fade_png):
            print(f"wrote {p.name} (install Pillow to verify pixel width)")


if __name__ == "__main__":
    main()
