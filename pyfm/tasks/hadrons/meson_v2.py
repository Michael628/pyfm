import typing as t

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.tasks.hadrons.meson import MesonConfig, get_incomplete_gammas, create_outfile_catalog

from pyfm import utils


def build_input_params(config: MesonConfig) -> HadronsInput:
    """Canonical-schema counterpart of ``meson.build_input_params``.

    Same grouping/skip-if-complete logic as ``meson.py:76-125`` — only the
    emitted modules differ: each group's gamma set is first published
    through a dedicated ``MFermion::SpinTaste`` module (``spin_taste``),
    then referenced by name from ``meson_field_v2`` (canonical
    ``MContraction::StagA2AMesonField``, SpinTaste-module-driven) instead
    of the inline ``spinTaste`` dict ``meson.py`` still emits for
    ``StagA2AMesonFieldLegacy``. No cross-group SpinTaste-module reuse is
    needed here (unlike ``highmode_v2``'s axial-gamma trick) — each group's
    gamma set is disjoint and self-contained, one module per group.
    """
    modules = {}
    schedule = []

    meson_template = config.meson.filestem

    bad_files = None
    if not config.overwrite:
        bad_files = utils.io.get_bad_files(create_outfile_catalog(config))

    for op_type, gammas in config.operations.group_by_mass_and_shift():
        assert len(op_type.mass) == 1, "Grouped operations should each have only 1 mass"
        op_label = op_type.gamma.name.lower()
        mass_label = op_type.mass[0]
        gauge = "" if op_type.gamma.local else config.shift_gauge_name

        if gauge is None:
            assert not op_type.gamma.local
            raise ValueError(
                "shift_gauge_name must be provided to meson config for non-local operations."
            )

        if not config.overwrite:
            gammas = get_incomplete_gammas(config, gammas, mass_label, bad_files)
            if not gammas:
                continue

        gamma_string = " ".join([x.gamma_string for x in gammas])

        output = meson_template.format(
            mass=config.mass.to_string(mass_label, remove_prefix=True)
        )

        module_name = f"mf_{op_label}_mass_{mass_label}"
        spintaste_name = f"spintaste_{module_name}"

        schedule.append(spintaste_name)
        modules[spintaste_name] = hadmods.spin_taste(
            name=spintaste_name,
            gammas=gamma_string,
            gauge=gauge,
            apply_g5=str(config.apply_g5).lower(),
        )

        schedule.append(module_name)
        modules[module_name] = hadmods.meson_field_v2(
            name=module_name,
            block=config.blocksize,
            gammas=spintaste_name,
            low_modes=config.low_modes_name.format(mass=mass_label),
            left=config.high_left_name,
            right=config.high_right_name,
            output=output,
        )

    return HadronsInput(modules=modules, schedule=schedule)
