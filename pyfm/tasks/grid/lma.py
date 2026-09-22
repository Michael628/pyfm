import typing as t

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
from pyfm.tasks.hadrons.types import SolveCrossTerms
import pyfm.tasks.grid.modules as gridmods


@dataclass(frozen=True)
class GridLMAConfig(CompositeConfig):
    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: meson.MesonConfig
    high_modes_config: t.List[HighModeConfig]
    series: str = ""
    skip_epack: bool = False
    skip_meson: bool = False
    skip_high_modes: bool = False

    solver_map: t.ClassVar[t.Dict[str, str]] = {"ranLL": "lma", "ama": "mpcg"}


def hadrons_to_grid_filestem(filestem: str, series: str) -> str:
    return filestem.removesuffix(f"_{series}").removesuffix("_t{tsource}")


def get_high_mode_run_tsources(config: HighModeConfig) -> t.List[str]:
    """Return high-mode source times that need Grid work generated.

    This mirrors the Hadrons high-mode overwrite criterion: when overwrite is
    disabled, rerun an entire source time if any expected output for that time
    source is missing. Existing-but-undersized files are intentionally not
    treated as missing here to match ``highmode.build_input_params``.
    """

    if config.overwrite:
        return list(map(str, config.tsource_range))

    df = highmode.create_outfile_catalog(config)
    if df.empty:
        return []
    missing_files = df[df["exists"] == False]
    return [
        str(tsource)
        for tsource in config.tsource_range
        if any(missing_files["tsource"] == str(tsource))
    ]


def build_a2a_params(config: meson.MesonConfig) -> t.List[t.Dict]:
    """Build Grid meson-field parameters, respecting meson overwrite logic."""

    bad_files = None
    if not config.overwrite:
        bad_files = utils.io.get_bad_files(meson.create_outfile_catalog(config))

    a2a = []
    for op, gammas in config.operations.group_by_mass_and_shift():
        assert len(op.mass) == 1, "Grouped operations should each have only 1 mass"
        mass_label = op.mass[0]

        if not config.overwrite:
            gammas = meson.get_incomplete_gammas(config, gammas, mass_label, bad_files)
            if not gammas:
                continue

        a2a.append(
            gridmods.meson_field(
                block=str(config.blocksize),
                mass=str(config.mass[mass_label]),
                output=config.meson.filestem.format(
                    mass=config.mass.to_string(mass_label, True)
                ),
                gammas=" ".join(g.gamma_string for g in gammas),
                apply_g5=str(config.apply_g5).lower(),
            )
        )

    return a2a


def build_input_params(config: GridLMAConfig) -> t.Dict:
    """Generate input parameters for the full LMA task.

    Orchestrates gauge module generation with submodule computation, ensuring that
    gauge action modules are generated only when needed by the submodules that use them.

    Grid supports a single high-mode entry (``validate_config`` enforces
    ``len(high_modes_config) <= 1``); the first entry is read guardedly so a
    skipped (empty) high-modes section still builds the unconditional
    ``mpcg`` block with the same defaults the old always-constructed sibling
    carried (residual ``1e-8`` -> ``"1e-08"``, mixed precision on).
    """

    hm = config.high_modes_config[0] if config.high_modes_config else None
    mpcg_residual = hm.residual[0] if hm is not None else 1e-8
    mpcg_mixed = (hm.solver == "mpcg") if hm is not None else True

    gauge = gridmods.gauge_files(
        link=config.gauge_config.ildg_links.filestem,
        fatlink=config.gauge_config.fat_links.filestem,
        longlink=config.gauge_config.long_links.filestem,
    )
    op_type = "load" if config.epack_config.load else "solve"
    irl_kwargs = (
        config.epack_config.lanczos.to_string() if not config.epack_config.load else {}
    )
    eigfile = (
        config.epack_config.eig.filestem
        if config.epack_config.load or config.epack_config.save_eigs
        else ""
    )
    evalfile = (
        config.epack_config.eval.filestem if config.epack_config.save_evals else ""
    )
    epack_params = gridmods.epack(
        op_type,
        eigfile,
        str(config.epack_config.eigs),
        evalfile,
        multifile=str(config.epack_config.multifile).lower(),
        mass="0.0" if not config.epack_config.load else None,
        **irl_kwargs,
    )
    corr = []
    highModeActions = []
    sources = []
    if not config.skip_high_modes and hm is not None:
        run_tsources = get_high_mode_run_tsources(hm)

        sources = [
            gridmods.random_wall_source(
                t_step=str(hm.time),
                t0=tsource,
                n_src=str(hm.noise),
                seed=f"noise_t{tsource}",
            )
            for tsource in run_tsources
        ]

        if run_tsources:
            highModeActions = [
                gridmods.action(m, str(hm.mass[m]))
                for m in hm.masses
            ]

            for op, con in contraction_gen(hm):
                solver_label = con.solver_label
                mass_label = con.mass_label(hm.mass)

                corr.append(
                    gridmods.contraction(
                        quark_solver=config.solver_map[con.quark.solver],
                        quark_action=con.quark.mass,
                        antiquark_solver=config.solver_map[con.antiquark.solver],
                        antiquark_action=con.antiquark.mass,
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
                        output=hadrons_to_grid_filestem(
                            hm.high_modes.filestem, config.series
                        ).format(
                            mass=mass_label,
                            dset=solver_label,
                            gamma_label=op.gamma.name.lower(),
                        ),
                    )
                )
    if config.skip_meson:
        a2a = []
    else:
        a2a = build_a2a_params(config.meson_config)

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
        mpcg=gridmods.mpcg(
            residual=str(mpcg_residual),
            mixed_precision=str(mpcg_mixed).lower(),
        ),
        **{k: v for k, v in optional.items() if v is not None},
    )


def validate_config(config: GridLMAConfig) -> None:
    """Validate GridLMAConfig after construction.

    Delegates to the shared LMI validator for cross-sibling concerns, then
    guards Grid capability: at most one high-mode entry (the Grid LMA path
    resolves a single source family through ``solver_map``, and the mpcg
    block pins one residual), and the **effective** solve-cross mode must be
    DIAGONAL — TIERED/ALL that collapse to DIAGONAL under skip flags emit
    only same-solver pairs and are valid Grid workloads.
    """
    lmi.validate_config(config)
    if len(config.high_modes_config) > 1:
        raise ValueError(
            f"grid_lma supports at most one high_modes_config entry; got "
            f"{len(config.high_modes_config)}. The Grid LMA path resolves a "
            "single source family (one mpcg residual; solver_map covers only "
            "ranLL and ama). Use the Hadrons LMI task for multi-entry "
            "workloads."
        )
    if not config.high_modes_config:
        return
    hm = config.high_modes_config[0]
    if hm.effective_solve_cross_terms != SolveCrossTerms.DIAGONAL:
        raise ValueError(
            "grid_lma only supports solve_cross_terms=DIAGONAL; got "
            f"{hm.solve_cross_terms!r} (effective "
            f"{hm.effective_solve_cross_terms!r}). Cross-solver contractions "
            "are not implemented for the Grid LMA path — use the Hadrons LMI "
            "task instead."
        )
    if len(hm.residual) > 1:
        raise ValueError(
            "grid_lma supports exactly one high-mode residual; got "
            f"{hm.residual}. solver_map resolves only ranLL and ama — "
            "per-residual ama_{r} labels have no Grid solver (and the mpcg "
            "block pins residual[0])."
        )

    if hm.low_mode_method == "load" and not hm.skip_low_modes:
        raise ValueError(
            "grid_lma does not support low_mode_method='load' (the "
            "file-driven StagLMAMesonFieldProp producer chain is "
            "Hadrons-only; grid's solver_map covers only ranLL and ama). "
            "Use the Hadrons LMI task for meson-field low modes."
        )


# Register GridLMAConfig with all handlers
register_task(
    "grid_lma",
    GridLMAConfig,
    lmi.create_outfile_catalog,
    build_input_params,
    lmi.build_aggregator_params,
    lmi.normalize_params,
    lmi.route_params,
    validate=validate_config,
)
