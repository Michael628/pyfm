import re
import logging
import os
import typing as t
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import pyfm.processing.processor as pc
import pandas as pd
from pydantic.dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass
class PlotDefaults:
    figsize: t.Tuple[int, int] = (7, 5)
    dpi: int = 200
    fontsize_big: int = 16
    fontsize: int = 14
    fontsize_small: int = 7
    xlabel: str = "$t/a$"
    ylabel: str = "$C(t)$"


def get_plot(**kwargs):
    return plt.subplots(figsize=PlotDefaults.figsize, dpi=PlotDefaults.dpi, **kwargs)


def plot_hl(x: t.Sequence, y: float, ax=None, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = get_plot()
    ax.plot(x, [y] * len(x), color="black", linestyle="--", linewidth=0.5)
    # ax.set_prop_cycle(None)

    return ax


def plot_data_hist(df: pd.DataFrame, kind: str = "violin", normalize: bool = True):
    norm_df = df
    if normalize:
        norm_df = df.groupby("label").apply(pc.norm_dist)

    return sns.catplot(
        data=norm_df,
        x="corr",
        kind=kind,
        col="t",
        hue="label",
        col_wrap=4,
        sharex=False,
        # whis=(0.3, 99.7),
        dodge=True,
    ).set_titles(norm_df.iloc[0]["gamma_label"] + " {col_var}={col_name}")


def plot_data_swarm(
    df: pd.DataFrame, ax, x: str, y: str, hue=None, inset: bool = False, **kwargs
) -> plt.Axes:
    sns.swarmplot(data=df, x=x, y=y, hue=hue, ax=ax)
    if inset:
        inset_loc = kwargs.get("inset_loc", "upper right")
        axins = inset_axes(ax, width="30%", height="30%", loc=inset_loc)
        sns.stripplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            ax=axins,
        )
        axins.legend().remove()
        axins.tick_params(
            axis="both",
            which="both",
            direction="in",
            labelsize=PlotDefaults.fontsize_small,
        )
        if inset_ylim := kwargs.get("inset_ylim", None):
            axins.set_ylim(inset_ylim)
            # axins.set_yticklabels(inset_ylim)
        if inset_xlim := kwargs.get("inset_xlim", None):
            axins.set_xlim(inset_xlim)
        axins.set_xlabel(None)
        axins.set_ylabel(None)

        if kwargs.get("inset_log", False):
            axins.semilogy()

    return ax


def plot_data_errorbar(df: pd.DataFrame, ax, abs=False, **kwargs) -> plt.Axes:
    offset = -0.2
    offset_frac = np.abs(2 * offset) / len(df.columns)
    for col in df.columns:
        x = np.array(df.index) + offset
        offset += offset_frac
        y = gv.mean(df[col].to_numpy())
        if abs:
            y = np.abs(y)
        yerr = gv.sdev(df[col].to_numpy())
        label = col
        if isinstance(label, t.Tuple):
            label = ", ".join([str(l) for l in label])
            label = f"({label})"
        ax.errorbar(
            x,
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
    return ax


def plot_cols(
    df: pd.DataFrame, plot_func=plot_data_errorbar, ax=None, **kwargs
) -> plt.Axes:
    def axis_setup(axis: plt.Axes) -> t.Tuple[plt.Axes, str]:
        axis.xaxis.set_tick_params(direction="in")
        axis.yaxis.set_tick_params(direction="in")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        axis.set_facecolor("white")
        axis.semilogy()

        if lattice_units := kwargs.get("afm", None):
            lattice_units = float(lattice_units)

            def convert_to_top_units(x):
                return x * lattice_units

            def convert_to_bottom_units(x):
                return x / lattice_units

            top_ax = axis.secondary_xaxis(
                "top", functions=(convert_to_top_units, convert_to_bottom_units)
            )
            top_ax.set_xlabel("$t (fm)$", fontsize=PlotDefaults.fontsize)

        if title := kwargs.get("title", None):
            axis.set_title(kwargs["title"], fontsize=PlotDefaults.fontsize_big)
        if xlabel := kwargs.get("xlabel", "$t/a$"):
            axis.set_xlabel(xlabel, fontsize=PlotDefaults.fontsize)
        if ylabel := kwargs.get("ylabel", "$C(t)$"):
            axis.set_ylabel(ylabel, fontsize=PlotDefaults.fontsize)
        if "xlim" in kwargs:
            axis.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            axis.set_ylim(kwargs["ylim"])

        legend_loc = kwargs.get("legend_loc", None)
        legend_title = kwargs.get("legend_title", None)
        if (legend_title is None) and (None not in df.columns.names):
            legend_title = ", ".join(df.columns.names)
            legend_title = f"({legend_title})"

        axis.legend(title=legend_title, loc=legend_loc)

        return axis, title

    if ax is None:
        fig, ax_out = get_plot()
    else:
        ax_out = ax

    ax_out = plot_func(df, ax_out, **kwargs)

    ax_out, title = axis_setup(ax_out)

    if figdir := kwargs.get("figdir", None):
        file_name = kwargs.get("file_name", None)
        file_name = file_name or title

        if file_name:
            file_name = file_name.lower()
            file_name = file_name.replace(" ", "_")
            file_name = re.sub(r"[^0-9a-z_-]", "", file_name)
        else:
            raise ValueError(
                "title or file_name required to save figure, please provide a title in the kwargs"
            )

        figdir = figdir.rstrip("/")
        figdir = figdir + "/"
        os.makedirs(figdir, exist_ok=True)

        plt.savefig(
            figdir + file_name + ".png",
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )

    return ax_out
