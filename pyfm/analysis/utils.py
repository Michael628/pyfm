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


def processed_df_gen(
    df: pd.DataFrame | t.Iterator,
    proc_fun: t.Callable,
    group_as_param: bool = False,
    **kwargs,
):
    if isinstance(df, pd.DataFrame):
        if group_as_param:
            yield None, proc_fun(df, None, **kwargs)
        else:
            yield None, proc_fun(df, **kwargs)
    else:
        for g, df_out in df:
            if group_as_param:
                yield g, proc_fun(df_out, g, **kwargs)
            else:
                yield g, proc_fun(df_out, **kwargs)


def collapse_iter(
    df: pd.DataFrame | t.Iterator,
) -> pd.DataFrame:
    return pd.concat([x for _, x in processed_df_gen(df, lambda x: x)])


def outlier_mask(
    df: pd.DataFrame, n_std: int = 3, filter_by_cfg=True, invert=True
) -> pd.DataFrame:
    mean = df.groupby("t")["corr"].transform("mean")
    std = df.groupby("t")["corr"].transform("std")
    cutoff = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    mask = (df["corr"] < lower) | (df["corr"] > upper)

    if filter_by_cfg:
        outlier_cfgs = df[mask]["series_cfg"].drop_duplicates().to_list()
        if len(outlier_cfgs) != 0:
            logging.debug(f"{len(outlier_cfgs)} outliers found")
            logging.debug(f"Outlier configs: {outlier_cfgs}")
        else:
            logging.debug(f"No outliers found.")

        if invert:
            return ~(df["series_cfg"].isin(outlier_cfgs))
        return df["series_cfg"].isin(outlier_cfgs)
    else:
        if invert:
            return ~mask
        else:
            return mask


def remove_outlier_gen(df: pd.DataFrame | t.Iterator, *args, **kwargs):
    """Iterates over df items removing outlier points

    Args:
        df: Dataframe or iterable to filter
        filter_by_cfg: If true, removes all points in df that share a config with outlier
        *args: See `outliers_std_mask`
        **kwargs: See `outliers_std_mask`

    Yields:
        NamedTuple describing group filter and filtered dataframe
    """

    def remove_outliers(df, group, *args, **kwargs):
        logging.debug(group)
        return df[outlier_mask(df, *args, **kwargs)]

    yield from processed_df_gen(
        df, remove_outliers, group_as_param=True, *args, **kwargs
    )


def lmi_data_gen(df: pd.DataFrame | t.Iterator):
    def calculate_lmi(df: pd.DataFrame) -> pd.DataFrame:
        def replace_low_modes(df: pd.DataFrame) -> pd.DataFrame:
            agg_dict = {k: "first" for k in df.columns} | {"corr": "mean"}

            return (
                df[df["dset"] == "ama"]
                .copy()
                .assign(
                    corr=(
                        df.loc[df.dset == "ama", "corr"].values
                        - df.loc[df.dset == "ranLL", "corr"].values
                        + df.loc[df.dset == "a2aLL", "corr"].values
                    )
                )
                .groupby(["series_cfg", "t"], as_index=False)
                .agg(agg_dict)
                .drop("gamma", axis=1)
                .assign(dset="lmi")
            )

        return replace_low_modes(df.sort_values(["series_cfg", "gamma", "t"]))

    for g, df_out in complete_dset_gen(df):
        yield g, calculate_lmi(df_out)


def plot_signal_noise_nts(df: pd.DataFrame | t.Iterator, **kwargs):
    def plotter(
        df: pd.DataFrame,
        fold=False,
        title: str = "",
        postproc_func: t.Callable = lambda x: x,
        **kwargs,
    ):
        gvar_data = (
            pc.signal_noise_nts(df, "corr", ["label"], fold=fold)
            .pipe(postproc_func)
            .assign(
                label=lambda x: x.label + " (" + x.ncfgs.astype("string") + " cfgs)"
            )
        )

        plot_params = {
            "nts": {"ylabel": f"$N/S$"},
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

        for gg, d in pc.masked_df_gen(gvar_data, pc.col_mask_gen(gvar_data, ["dset"])):
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
                plot.plot_hl(pivot.index, 1.0, ax)

    plotter_wrap = functools.partial(plotter, **kwargs)
    for _ in processed_df_gen(df, plotter_wrap):
        pass


def complete_dset_mask(df: pd.DataFrame, expected_dsets: int = 3) -> pd.Series:
    mask = df.groupby("series_cfg")["dset"].nunique() == expected_dsets
    # NOTE: how it's done when series_cfg is in index
    # mask = mask.loc[df.index.get_level_values("series_cfg")].values

    incomplete_configs = mask[~mask].index.values

    if len(incomplete_configs) != 0:
        logging.debug(f"Incomplete configs: {incomplete_configs}")
    else:
        logging.debug("`complete_dset_mask`: No configs filtered.")

    mask = mask.loc[df["series_cfg"]].values
    if len(df[mask]) == 0:
        raise ValueError(f"No configs found with {expected_dsets} data sets")
    return mask


def complete_dset_gen(df: pd.DataFrame | t.Iterator, invert=False, *args, **kwargs):
    for g, df_out in processed_df_gen(df, lambda x: x):
        logging.debug(g)
        mask = complete_dset_mask(df_out, *args, **kwargs)
        if invert:
            mask = ~mask
        yield g, df_out[mask]


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
    def printer(df, group) -> None:
        if group:
            print(f"{group}")
        outlier_mask = outliers_std_mask(df, **kwargs)
        outliers = t.cast(pd.DataFrame, df[outlier_mask])
        if len(outliers) != 0:
            outlier_cfgs = outliers["series_cfg"].drop_duplicates().to_list()
            print(f"{len(outlier_cfgs)} outliers found")
            print(f"Outlier configs: {outlier_cfgs}")
        else:
            print(f"No outliers found.")
        print()

    for group, p_func in processed_df_gen(df, printer, group_as_param=True):
        pass


def show_strip_plot(
    df: pd.DataFrame | t.Iterator, preproc_func: t.Callable = lambda x: x, **kwargs
) -> None:
    def plotter(df: pd.DataFrame, group: t.NamedTuple) -> None:
        plot.plot_data_hist(preproc_func(df), kind="strip")
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

    for _ in processed_df_gen(df, plotter, group_as_param=True):
        pass
