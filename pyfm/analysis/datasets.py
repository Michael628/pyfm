import pandas as pd
import typing as t
from pyfm.analysis.domain import WrappedDataPipe
import pyfm.processing.processor as pc
import pyfm.processing.dataio as dio

from .services.filter import create_col_iter, filter_outliers, filter_incomplete_lmi
from .services.process import append_lmi, average


def fix_sign_flip(df):
    def flip(x):
        if x.iloc[0] < 0:
            x = -x
        return x

    subsets = [c for c in df.columns if c != "corr"]
    subset_mask_iter = pc.generate_column_masks(df, subsets)
    for _, subset_mask in subset_mask_iter:
        df.loc[subset_mask, "corr"] = (
            df[subset_mask].groupby("series_cfg")["corr"].transform(flip)
        )
    return df


def strip_vals(d: dict) -> list:
    return list(d.values())


def post_processing(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(fix_sign_flip)
        .reset_index()
        .convert_dtypes(dtype_backend="numpy_nullable")
    )


def load_and_clean(fstring: str) -> WrappedDataPipe:
    return (
        dio.load_files_new(fstring, wildcard_fill=True)
        .pipe(lambda x: x.drop(columns="index") if "index" in x.columns else x)
        .pipe(post_processing)
        .pipe(lambda x: x.assign(gamma="G5") if "gamma" not in x.columns else x)
    )


def clean_lmi(df: pd.DataFrame, n_std=5) -> WrappedDataPipe:
    def clean_outliers(f: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [filter_outliers(x, n_std).passed for _, x in f.groupby("dset")]
        )

    return (
        create_col_iter(df, ["gamma_label"])
        .pipe(lambda x: filter_incomplete_lmi(x).passed)
        .pipe(append_lmi)
        .pipe(average)
        .pipe(clean_outliers)
        .nop(lambda x: print(x["series_cfg"].nunique()))
        # .nop(lambda x: print(x))
    )


async def load_data(
    fstring: str,
    post_proc_fn: t.Callable | None = None,
    repl: t.Dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    data = await dio.load(
        fstring,
        replacements=repl if repl is not None else {},
        wildcard_fill=True,
        **kwargs,
    ).pipe(post_proc_fn if post_proc_fn is not None else lambda x: x)

    return data


async def l144288_2keig_perlmutter() -> pd.DataFrame:
    """Load data generated from ERCAP 2023/2024 on Perlmutter

    Returns:
        pd.DataFrame with extra columns `ext` = 'perlmutter', `eigs` = '2000'
    """

    fstring = "~/Dropbox/2-Areas/lattice_data/l144288/e{eigs}n1dt2_{ext}/m000569/{gamma_label}/{dset}/corr_{gamma_label}_{dset}_m000569.h5"
    replacements = {"ext": ["perlmutter"], "eigs": "2000", "mass": "000569"}
    return (await load_data(fstring, post_processing, replacements)).assign(
        label="2k_eig_p"
    )


async def l144288_2keig_frontier() -> pd.DataFrame:
    """Load data generated from ALCC/INCITE 2023-2024 on Frontier

    Returns:
        pd.DataFrame with extra columns `ext` = 'frontier', `eigs` = '2000'
    """

    fstring = "~/Dropbox/2-Areas/lattice_data/l144288/e{eigs}n1dt2_{ext}/m000569/{gamma_label}/{dset}/corr_{gamma_label}_{dset}_m000569.h5"
    replacements = {"ext": ["frontier"], "eigs": "2000", "mass": "000569"}

    return (await load_data(fstring, post_processing, replacements)).assign(
        label="2k_eig_f"
    )


async def l144288_2keig():
    return pd.concat(
        [await l144288_2keig_frontier(), await l144288_2keig_perlmutter()]
    ).assign(label="2k_eig")


async def l144288_4keig() -> pd.DataFrame:
    """Load 4k eigenvector data. This is a combination of three data sets
    1. 4k eigs Data generated for ALCC 2025 on Frontier with the full ama, ranLL, a2aLL stack
        (`ext` = 'full', `eigs` = '4000')
    2. 4k eigs Data generated for ALCC 2025 on Frontier with projections only, i.e. ranLL and a2aLL
        (`ext` = 'proj', `eigs` = '4000')
    3. High mode data (ama) generated with 2k eigs in 2023/2024 restricted to configs found in dset (2.)
        (`ext` = 'orig', `eigs` = '2000')

    Returns:
        pd.DataFrame with extra columns `ext` = 'proj', `eigs` = '2000'
    """
    fstring = "~/Dropbox/2-Areas/lattice_data/l144288/e{eigs}n1dt2_{ext}/processed_dataframes/m000569/{gamma_label}/{dset}/corr_{gamma_label}_{dset}_m000569.h5"

    replacements = {"ext": ["proj", "full"], "eigs": "4000", "mass": "000569"}
    data_4k = (await load_data(fstring, post_processing, replacements)).query(
        'series_cfg not in ["a.96", "a.108"]'
    )  # Remove duplicate configs. These were generated both by 'full' and by 'proj'

    data_2k = await l144288_2keig()

    proj_cfgs = data_4k[data_4k["ext"] == "proj"]["series_cfg"].unique()
    mask = data_2k["dset"] == "ama"
    mask &= data_2k["series_cfg"].isin(proj_cfgs)
    data_ama = data_2k[mask].copy().assign(ext="proj", eigs="4000")

    return pd.concat([data_4k, data_ama]).assign(label="4k_eig")
