import pandas as pd
import numpy as np
import typing as t
from itertools import islice
import pyfm.processing.processor as pc
from pyfm.processing import plot
import functools
import re
import os
import matplotlib.pyplot as plt
import logging


def generate_processed_df(
    df: pd.DataFrame | t.Iterator,
    proc_fn: t.Callable,
    group_as_param: bool = False,
    **kwargs,
):
    if isinstance(df, pd.DataFrame):
        if group_as_param:
            yield None, proc_fn(df, None, **kwargs)
        else:
            yield None, proc_fn(df, **kwargs)
    else:
        for g, df_out in df:
            logging.debug(f"group: {g}")
            if group_as_param:
                yield g, proc_fn(df_out, g, **kwargs)
            else:
                yield g, proc_fn(df_out, **kwargs)


def generate_masked_df(
    df: pd.DataFrame | t.Iterator, mask_fn: t.Callable, *args, **kwargs
):
    def apply_mask(df, *args, **kwargs):
        return df[mask_fn(df, *args, **kwargs)]

    yield from generate_processed_df(df, apply_mask, *args, **kwargs)


def collapse_iter(
    df: pd.DataFrame | t.Iterator,
) -> pd.DataFrame:
    return pd.concat([x for _, x in generate_processed_df(df, lambda x: x)])


def plot_signal_noise_nts(df: pd.DataFrame | t.Iterator, **kwargs):
    def plotter(
        df: pd.DataFrame,
        fold=False,
        title: str = "",
        postproc_fn: t.Callable = lambda x: x,
        **kwargs,
    ):
        gvar_data = (
            pc.signal_noise_nts(df, "corr", ["label"], fold=fold)
            .pipe(postproc_fn)
            .assign(
                label=lambda x: x.label + " (" + x.ncfgs.astype("string") + " cfgs)"
            )
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

    plotter_wrap = functools.partial(plotter, **kwargs)
    for _ in generate_processed_df(df, plotter_wrap):
        pass


def show_config_range(a: pd.DataFrame, cfgnos: bool = True) -> None:
    series_cfg = (
        a["series_cfg"]
        .drop_duplicates()
        .str.split(".", expand=True)
        .rename(columns={0: "series", 1: "cfg"})
        .astype({"series": pd.StringDtype(), "cfg": pd.UInt16Dtype()})
    )

    def elem_gen(it, chunk_size):
        while True:
            if chunk := tuple(islice(it, chunk_size)):
                yield chunk
            else:
                break

    print(f"Total Configurations: {len(series_cfg)}\n")
    for group, val in series_cfg.groupby("series")["cfg"]:
        col_len = len(val) if len(val) <= 4 else 4
        min_val = val.min()
        max_val = val.max()
        count = len(val)
        avg_sep = (max_val - min_val) / count
        print(
            (
                f"\nseries `{group}` : {count} configurations\n"
                f"min: {min_val}, max: {max_val}\navg separation: {avg_sep:0.2f}\n"
            ),
        )
        if cfgnos:
            item_iter = iter(sorted(val.to_list()))
            items = [("{:5d} " * len(i)).format(*i) for i in elem_gen(item_iter, 4)]
            print("\n".join(items))


def print_outliers(df: pd.DataFrame | t.Iterator, **kwargs) -> None:
    def printer(df) -> None:
        outliers = next(remove_outliers(df, **kwargs))[1]
        print(outliers)
        if len(outliers) != 0:
            outlier_cfgs = outliers["series_cfg"].drop_duplicates().to_list()
            print(f"{len(outlier_cfgs)} outliers found")
            print(f"Outlier configs: {outlier_cfgs}")
        else:
            print(f"No outliers found.")
        print()

    for group, p_fn in generate_processed_df(df, printer):
        pass


def show_strip_plot(
    df: pd.DataFrame | t.Iterator,
    preproc_fn: t.Callable = lambda x: x,
    normalize: bool = True,
    **kwargs,
) -> None:
    def plotter(df: pd.DataFrame, group: t.NamedTuple) -> None:
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

    for _ in generate_processed_df(df, plotter, group_as_param=True):
        pass
