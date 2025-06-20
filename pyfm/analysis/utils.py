import pandas as pd
import numpy as np
import typing as t
from itertools import islice
import pyfm.processing.processor as pc
from pyfm.processing import plot
import functools


def outliers_std_mask(df: pd.DataFrame, n_std: int = 3) -> pd.DataFrame:
    mean = df.groupby("t")["corr"].transform("mean")
    std = df.groupby("t")["corr"].transform("std")
    cutoff = std * n_std
    lower, upper = mean - cutoff, mean + cutoff
    outlier_mask = (df["corr"] < lower) | (df["corr"] > upper)
    return outlier_mask


def outliers_cfg_mask(df: pd.DataFrame, *args, **kwargs):
    mask = outliers_std_mask(df, *args, **kwargs)

    outlier_cfgs = df[mask]["series_cfg"].drop_duplicates()
    return df["series_cfg"].isin(outlier_cfgs)


def processed_df_gen(df: pd.DataFrame | t.Iterator, proc_fun: t.Callable):
    if isinstance(df, pd.DataFrame):
        yield None, proc_fun(df)
    else:
        for g, df_out in df:
            yield g, proc_fun(df_out)


def remove_outlier_gen(df: pd.DataFrame | t.Iterator, invert=False, *args, **kwargs):
    for g, df_out in processed_df_gen(df, lambda x: x):
        mask = ~outliers_cfg_mask(df_out, *args, **kwargs)
        if invert:
            mask = ~mask
        yield g, df_out[mask]


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

    yield from processed_df_gen(df, calculate_lmi)


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
    mask = mask.loc[df["series_cfg"]].values
    if len(df[mask]) == 0:
        raise ValueError(f"No configs found with {expected_dsets} data sets")
    return mask


def complete_dset_gen(df: pd.DataFrame | t.Iterator, invert=False, *args, **kwargs):
    for g, df_out in processed_df_gen(df, lambda x: x):
        mask = complete_dset_mask(df_out, *args, **kwargs)
        if invert:
            mask = ~mask
        yield g, df_out[mask]


def get_config_range(a: pd.DataFrame) -> None:
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
                f"series `{group}` : {count} configurations\n"
                f"min: {min_val}, max: {max_val}\navg separation: {avg_sep:0.2f}\n"
            ),
        )
        item_iter = iter(sorted(val.to_list()))
        items = [("{:5d} " * len(i)).format(*i) for i in elem_gen(item_iter, 4)]
        print("\n".join(items))


def print_outliers(df: pd.DataFrame | t.Iterator, **kwargs) -> None:
    def printer(group, df) -> None:
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

    for group, p_func in processed_df_gen(
        df, lambda x: functools.partial(printer, df=x)
    ):
        p_func(group)


def show_strip_plot(
    df: pd.DataFrame | t.Iterator, preproc_func: t.Callable = lambda x: x
) -> None:
    def plotter(df: pd.DataFrame) -> None:
        print(df)
        plot.plot_data_hist(preproc_func(df), kind="strip")

    for _ in processed_df_gen(df, plotter):
        pass
