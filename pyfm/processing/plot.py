import re
import logging
import os
import typing as t
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import pandas as pd
from pydantic.dataclasses import dataclass
from functools import partial
from pyfm.processing import processor

@dataclass
class PlotDefaults:
    figsize: t.Tuple[int, int] = (7, 5)
    dpi: int = 200
    fontsize_big: int = 16
    fontsize: int = 14
    xlabel: str = "$t/a$"
    ylabel: str = "$C(t)$"


def plot_cols(df: pd.DataFrame, abs=False, **kwargs) -> None:
    fig, ax = plt.subplots(
        figsize=PlotDefaults.figsize, dpi=PlotDefaults.dpi, ncols=1, nrows=1
    )

    for col in df.columns:
        y = gv.mean(df[col].to_numpy())
        if abs:
            y = np.abs(y)
        yerr = gv.sdev(df[col].to_numpy())
        label = col
        if isinstance(label, t.Tuple):
            label = "-".join([str(l) for l in label])
        ax.errorbar(
            df.index,
            y,
            yerr=yerr,
            marker="o",
            linestyle="None",
            # fillstyle="none",
            alpha=0.6,
            capsize=7,
            markeredgewidth=0.5,
            elinewidth=0.7,
            markersize=7,
            label=label,
        )

    ax.xaxis.set_tick_params(direction="in")
    ax.yaxis.set_tick_params(direction="in")

    ax.set_facecolor("white")
    ax.semilogy()

    ax.legend(title=kwargs.get("legend_title", None))

    if lattice_units := kwargs.get("afm", None):
        lattice_units = float(lattice_units)

        def convert_to_top_units(x):
            return x * lattice_units

        def convert_to_bottom_units(x):
            return x / lattice_units

        top_ax = ax.secondary_xaxis(
            "top", functions=(convert_to_top_units, convert_to_bottom_units)
        )
        top_ax.set_xlabel("$t (fm)$", fontsize=PlotDefaults.fontsize)

    if title := kwargs.get("title", None):
        ax.set_title(kwargs["title"], fontsize=PlotDefaults.fontsize_big)
    if xlabel := kwargs.get("xlabel", "$t/a$"):
        ax.set_xlabel(xlabel, fontsize=PlotDefaults.fontsize)
    if ylabel := kwargs.get("ylabel", "$C(t)$"):
        ax.set_ylabel(ylabel, fontsize=PlotDefaults.fontsize)
    if "xlim" in kwargs:
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])

    if figdir := kwargs.get("figdir", None):
        figdir = figdir.rstrip("/")
        figdir = figdir + "/"
        os.makedirs(figdir, exist_ok=True)
        if title:
            escape_title = title.lower()
            escape_title = escape_title.replace(" ", "_")
            escape_title = re.sub(r"[^0-9a-z_-]", "", escape_title)
            plt.savefig(
                figdir + escape_title + ".png",
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )
        else:
            logging.warning("Title required to save figure")

    plt.show()

def plot_cols_nts(df: pd.DataFrame, *args, **kwargs) -> None:
        processor.noise_nts
plot_cols_nts = functools.partial()(signal: pd.DataFrame, noise: t.Optional[pd.DataFrame] = None, **kwargs) -> None:
    if noise:


