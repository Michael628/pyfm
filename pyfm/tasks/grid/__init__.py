import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import CompositeConfig
from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons import (
    gauge,
    meson,
    epack,
    highmode,
    HighModeConfig,
    lmi,
)
from pyfm.tasks.hadrons.highmode.twopoint import contraction_gen
import pyfm.tasks.grid.modules as gridmods


@dataclass(frozen=True)
class GridLMIConfig(CompositeConfig):
    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: meson.MesonConfig
    high_modes_config: HighModeConfig
    series: str
    skip_epack: bool = False
    skip_meson: bool = False
    skip_high_modes: bool = False

    key: t.ClassVar[str] = "grid"


def hadrons_to_grid_filestem(filestem: str, series: str) -> str:
    return filestem.removesuffix(f"_{series}").removesuffix("_t{tsource}")


def build_input_params(config: GridLMIConfig) -> t.Dict:
    """Generate input parameters for the full LMI task.

    Orchestrates gauge module generation with submodule computation, ensuring that
    gauge action modules are generated only when needed by the submodules that use them.
    """

    gauge = gridmods.gauge_files(
        link=config.gauge_config.gauge_links.filestem,
        fatlink=config.gauge_config.fat_links.filestem,
        longlink=config.gauge_config.long_links.filestem,
    )
    op_type = "load" if config.epack_config.load else "solve"
    irl_kwargs = (
        config.epack_config.lanczos.to_string() if not config.epack_config.load else {}
    )
    epack_params = gridmods.epack(
        op_type,
        config.epack_config.eig.filestem,
        str(config.epack_config.eigs),
        config.epack_config.eval.filestem,
        multifile=str(config.epack_config.multifile).lower(),
        mass="0.0" if not config.epack_config.load else None,
        **irl_kwargs,
    )
    corr = []
    highModeActions = []
    sources = []
    if not config.skip_high_modes:
        highModeActions = [
            gridmods.action(str(config.high_modes_config.mass[m]))
            for m in config.high_modes_config.masses
        ]

        sources = [
            gridmods.random_wall_source(
                t_step=str(config.high_modes_config.time),
                t0=str(t),
                n_src=str(config.high_modes_config.noise),
                seed=f"noise_t{t}",
            )
            for t in range(
                config.high_modes_config.tstart,
                config.high_modes_config.tstop + 1,
                config.high_modes_config.dt,
            )
        ]

        for op, con in contraction_gen(config.high_modes_config):
            mass_label = con.mass_label(config.high_modes_config.mass)
            corr.append(
                gridmods.contraction(
                    quark=gridmods.spin_taste(
                        gammas=con.quark.gamma.gamma_string,
                        apply_g5=str(con.quark.apply_g5).lower(),
                    ),
                    antiquark=gridmods.spin_taste(
                        gammas=con.antiquark.gamma.gamma_string,
                        apply_g5=str(con.antiquark.apply_g5).lower(),
                    ),
                    sink=gridmods.spin_taste(
                        gammas=con.sink.gamma.gamma_string,
                        apply_g5=str(con.sink.apply_g5).lower(),
                    ),
                    lma_output=hadrons_to_grid_filestem(
                        config.high_modes_config.high_modes.filestem, config.series
                    ).format(
                        mass=mass_label,
                        dset="ranLL",
                        gamma_label=op.gamma.name.lower(),
                    ),
                    ama_output=hadrons_to_grid_filestem(
                        config.high_modes_config.high_modes.filestem, config.series
                    ).format(
                        mass=mass_label,
                        dset="ama",
                        gamma_label=op.gamma.name.lower(),
                    ),
                )
            )
    if config.skip_meson:
        a2a = []
    else:
        a2a = [
            gridmods.meson_field(
                block=str(config.meson_config.blocksize),
                mass=str(config.meson_config.mass[op.mass[0]]),
                output=config.meson_config.meson.filestem.format(
                    mass=config.high_modes_config.mass.to_string(op.mass[0], True)
                ),
                gammas=" ".join(g.gamma_string for g in gammas),
                apply_g5=str(config.meson_config.apply_g5).lower(),
            )
            for op, gammas in config.meson_config.operations.group_by_mass_and_shift()
        ]

    optional = dict(
        sources=dict(elem=sources) if sources else None,
        corr=dict(elem=corr) if corr else None,
        a2a=dict(elem=a2a) if a2a else None,
        highModeActions=dict(elem=highModeActions) if highModeActions else None,
    )
    return dict(
        gauge=gauge,
        epack=epack_params,
        lma=gridmods.lma(),
        mpcg=gridmods.mpcg(),
        **{k: v for k, v in optional.items() if v is not None},
    )


# Register GridLMIConfig with all handlers
register_task(
    GridLMIConfig,
    lmi.create_outfile_catalog,
    build_input_params,
    lmi.build_aggregator_params,
    lmi.preprocess_params,
    validate=lmi.validate_config,
)

__all__ = ["gridmods"]
