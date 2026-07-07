"""Publication/industrial figure conventions for operando-EIS analysis.

A single, importable source of the figure spec so every plot in a project comes
out identically sized and styled. The rules encoded here are deliberately
generic (no dataset, no study, no lab): tick-label size, uniform line width,
capacity axes anchored at zero, and a pixel-exact export width derived from a
target vector-art width.

Export-width rule
-----------------
``EXPORT_W_PX = TARGET_AI_WIDTH_PX * DPI_EXPORT / 72``

A vector-art canvas is specified in points (1 pt = 1/72 inch). To rasterize a
figure that will sit at ``TARGET_AI_WIDTH_PX`` points in a layout at
``DPI_EXPORT`` dots-per-inch without resampling, the bitmap must be exactly
``TARGET_AI_WIDTH_PX * DPI_EXPORT / 72`` pixels wide. :func:`export_width_px`
returns that integer; :func:`save_figure` guarantees the written PNG matches it.

Typical use
-----------
>>> from oeis_toolkit.figstyle import FigureSpec, apply_style, save_figure
>>> spec = FigureSpec()                       # defaults below
>>> import matplotlib.pyplot as plt
>>> apply_style(spec)
>>> fig, ax = plt.subplots(figsize=spec.figsize_in())
>>> ax.plot([0, 1, 2], [0, 1, 4])
>>> zero_floor(ax, axis="y")                   # capacity/quantity axes start at 0
>>> save_figure(fig, "demo.png", spec)         # writes an EXPORT_W_PX-wide PNG
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "FigureSpec",
    "export_width_px",
    "apply_style",
    "save_figure",
    "zero_floor",
]


@dataclass(frozen=True)
class FigureSpec:
    """Immutable figure-style parameters.

    Parameters
    ----------
    tick_fontsize : int
        Font size (pt) for tick labels. Default 12.
    label_fontsize : int
        Font size (pt) for axis labels. Default 13.
    title_fontsize : int
        Font size (pt) for the axes title. Default 13.
    linewidth : float
        Uniform data line width (pt). Default 1.8.
    dpi_export : int
        Raster export resolution (dots per inch). Default 300.
    target_ai_width_px : int
        Target vector-art width in points; drives the export pixel width.
        Default 528.
    aspect : float
        Height / width ratio for the default figure size. Default 9/16.
    """

    tick_fontsize: int = 12
    label_fontsize: int = 13
    title_fontsize: int = 13
    linewidth: float = 1.8
    dpi_export: int = 300
    target_ai_width_px: int = 528
    aspect: float = 9.0 / 16.0

    def __post_init__(self) -> None:
        for name in ("tick_fontsize", "label_fontsize", "title_fontsize",
                     "dpi_export", "target_ai_width_px"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if not (self.linewidth > 0):
            raise ValueError(f"linewidth must be positive, got {self.linewidth!r}")
        if not (self.aspect > 0):
            raise ValueError(f"aspect must be positive, got {self.aspect!r}")

    def export_width_px(self) -> int:
        """Exact raster width in pixels: ``target_ai_width_px * dpi_export / 72``."""
        return round(self.target_ai_width_px * self.dpi_export / 72)

    def figsize_in(self) -> tuple[float, float]:
        """Figure ``(width, height)`` in inches whose raster width equals
        :meth:`export_width_px` at ``dpi_export``."""
        width_in = self.export_width_px() / self.dpi_export
        return (width_in, width_in * self.aspect)


def export_width_px(spec: FigureSpec | None = None) -> int:
    """Return the export pixel width for ``spec`` (or the default spec)."""
    return (spec or FigureSpec()).export_width_px()


def apply_style(spec: FigureSpec | None = None) -> dict:
    """Apply the spec to Matplotlib's global rcParams and return the patch.

    Importing Matplotlib is deferred so that non-plotting code paths (and the
    unattended pipeline) never pay the import cost or require the dependency.
    """
    import matplotlib as mpl

    spec = spec or FigureSpec()
    patch = {
        "xtick.labelsize": spec.tick_fontsize,
        "ytick.labelsize": spec.tick_fontsize,
        "axes.labelsize": spec.label_fontsize,
        "axes.titlesize": spec.title_fontsize,
        "lines.linewidth": spec.linewidth,
        "figure.dpi": spec.dpi_export,
        "savefig.dpi": spec.dpi_export,
    }
    mpl.rcParams.update(patch)
    return patch


def zero_floor(ax, axis: Literal["x", "y", "both"] = "y") -> None:
    """Anchor the given axis(es) at zero without changing the upper limit.

    Capacity — and any non-negative quantity — should read against a zero
    baseline so bar heights and curve areas are visually honest.
    """
    if axis in ("y", "both"):
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top if top > 0 else None)
    if axis in ("x", "both"):
        right = ax.get_xlim()[1]
        ax.set_xlim(0, right if right > 0 else None)


def save_figure(fig, path: str | Path, spec: FigureSpec | None = None) -> Path:
    """Save ``fig`` as a PNG exactly :meth:`FigureSpec.export_width_px` wide.

    The figure width (in inches) is set so that ``width_in * dpi_export`` equals
    the target pixel width, then written at ``dpi_export``. Height is preserved
    by scaling both dimensions by the same factor, so aspect ratio is untouched.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    spec = spec or FigureSpec()
    target_px = spec.export_width_px()
    cur_w_in, cur_h_in = fig.get_size_inches()
    if cur_w_in <= 0:
        raise ValueError("figure has non-positive width")
    target_w_in = target_px / spec.dpi_export
    scale = target_w_in / cur_w_in
    fig.set_size_inches(cur_w_in * scale, cur_h_in * scale)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=spec.dpi_export)
    return out
