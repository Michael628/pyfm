import os
import typing as t
import pandas as pd
import matplotlib.pyplot as plt
import re

import pyfm.processing.processor as pc
import pyfm.processing.plot as plot


def plot_signal_noise_nts(
    df: pd.DataFrame,
    fold=False,
    title: str = "",
    postproc_fn: t.Callable = lambda x: x,
    **kwargs,
):
    gvar_data = (
        pc.signal_noise_nts(df, "corr", ["label"], fold=fold)
        .pipe(postproc_fn)
        .assign(label=lambda x: x.label + " (" + x.ncfgs.astype("string") + " cfgs)")
    )

    plot_params = {
        "nts": {"ylabel": "$N/S$"},
        "signal": {"ylabel": "$C(t)$"},
        "noise": {"ylabel": r"$\sigma(t)$"},
    }
    print(plot_params)
    print(kwargs)
    if all([k in plot_params for k in kwargs.keys()]):
        plot_params |= {k: plot_params[k] | v for k, v in kwargs.items()}
    else:
        plot_params = {k: v | kwargs for k, v in plot_params.items()}
    print(plot_params)

    for gg, d in pc.generate_column_dfs(gvar_data, ["dset"]):
        assert hasattr(gg, "dset") and gg is not None
        legend_title = title + f" {gg.dset}"
        pivot = pd.pivot_table(
            d,
            aggfunc="first",
            values="corr",
            index=["t"],
            columns=["label"],
        )
        ax = plot.plot_cols(
            pivot, afm=0.042, legend_title=legend_title, **plot_params[gg.dset]
        )
        if gg.dset == "nts":
            plot.plot_hl(pivot.index.to_list(), 1.0, ax)


def show_strip_plot(
    df: pd.DataFrame,
    group: t.NamedTuple,
    preproc_fn: t.Callable = lambda x: x,
    normalize: bool = True,
    **kwargs,
) -> None:
    plot.plot_data_hist(preproc_fn(df), kind="strip", normalize=normalize)
    if figdir := kwargs.get("figdir", None):
        file_name = kwargs.get("file_name", None).format(**group._asdict())

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

        print(figdir)
        print(file_name)
        plt.savefig(
            figdir + file_name + ".png",
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
