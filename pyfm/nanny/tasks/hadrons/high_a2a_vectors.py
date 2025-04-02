# This file generates xml parameters for the HadronsMILC app.
# Tasks performed:
#
# 1: Load eigenvectors
# 2: Generate noise sources
# 3: Solve low-mode propagation of sources
# 4: Solve CG on result of step 3
# 5: Subtract 3 from 4
# 6: Save result of 5 to disk

import logging
import typing as t

from pydantic.dataclasses import dataclass

from python_scripts.nanny import TaskBase
from python_scripts.nanny.config import OutfileList
from python_scripts.nanny.tasks.hadrons import SubmitHadronsConfig, templates


@dataclass
class A2AVecTask(TaskBase):
    mass: str
    subtract: bool
    epack: bool = True
    free: bool = False
    nstart: int = 0


def input_params(tasks: A2AVecTask, submit_config: SubmitHadronsConfig, outfile_config_list: OutfileList) -> t.Tuple[t.List[t.Dict], t.Optional[t.List[str]]]:
    submit_conf_dict = submit_config.string_dict()

    run_tsources = list(map(str, submit_config.tsource_range))


    if tasks.free:
        modules = [
            templates.unit_gauge('gauge'),
            templates.unit_gauge('gauge_fat'),
            templates.unit_gauge('gauge_long'),
            templates.cast_gauge('gauge_fatf', 'gauge_fat'),
            templates.cast_gauge('gauge_longf', 'gauge_long')
        ]
    else:
        gauge_filepath = outfile_config_list.gauge_links.filestem.format(**submit_conf_dict)
        gauge_fat_filepath = outfile_config_list.fat_links.filestem.format(**submit_conf_dict)
        gauge_long_filepath = outfile_config_list.long_links.filestem.format(**submit_conf_dict)

        modules = [
            templates.load_gauge('gauge', gauge_filepath),
            templates.load_gauge('gauge_fat', gauge_fat_filepath),
            templates.load_gauge('gauge_long', gauge_long_filepath),
            templates.cast_gauge('gauge_fatf', 'gauge_fat'),
            templates.cast_gauge('gauge_longf', 'gauge_long')
        ]

    if tasks.epack:
        epack_path = outfile_config_list.eig.filestem.format(**submit_conf_dict)

        # Load eigenvectors
        modules.append(templates.epack_load(name='epack',
                                            filestem=epack_path,
                                            size=submit_conf_dict['sourceeigs'],
                                            multifile=submit_conf_dict['multifile']))

        modules.append(templates.epack_modify(name=f"evecs_mass_{mass_label}",
                                              eigen_pack='epack',
                                              mass=mass))

    mass_label = tasks.mass
    name = f"stag_mass_{mass_label}"
    mass = str(submit_config.mass[mass_label])
    modules.append(templates.action(name=name,
                                    mass=mass,
                                    gauge_fat='gauge_fat',
                                    gauge_long='gauge_long'))

    name = f"istag_mass_{mass_label}"
    modules.append(templates.action_float(name=name,
                                          mass=mass,
                                          gauge_fat='gauge_fatf',
                                          gauge_long='gauge_longf'))

    dsets = ['ama']
    if tasks.epack:
        dsets.append('ranLL')
        modules.append(templates.lma_solver(
            name=f"stag_ranLL_mass_{mass_label}",
            action=f"stag_mass_{mass_label}",
            low_modes=f"evecs_mass_{mass_label}"
        ))

    modules.append(templates.mixed_precision_cg(
        name=f"stag_ama_mass_{mass_label}",
        outer_action=f"stag_mass_{mass_label}",
        inner_action=f"istag_mass_{mass_label}",
        residual='1e-8'
    ))

    vec_path = outfile_config_list.a2a_vec.filestem

    for seed_index in range(tasks.nstart,tasks.nstart+submit_config.noise):
        modules.append(templates.time_diluted_noise(f'w{seed_index}', 1))

        for slabel in dsets:
            quark = f"quark_{slabel}_mass_{mass_label}_n{seed_index}"
            source = f"w{seed_index}_vec"
            solver = f"stag_{slabel}_mass_{mass_label}"
            guess = ""
            if slabel == "ama":
                if tasks.subtract:
                    solver += "_subtract"
                if tasks.epack:
                    guess = f"quark_ranLL_mass_{mass_label}_n{seed_index}"

            modules.append(templates.quark_prop(
                name=quark,
                source=source,
                solver=solver,
                guess=guess,
                gammas='',
                apply_g5='false',
                gauge=""
            ))

            if slabel == 'ama':
                output = vec_path.format(mass=submit_config.mass_out_label[mass_label],
                                         seed_index=str(seed_index),
                                         **submit_conf_dict)
                modules.append(templates.save_vector(f'{quark}_save', quark, output, 'true'))

    schedule = [m["id"]['name'] for m in modules]

    return modules, schedule


def bad_files(task_config: TaskBase, submit_config: SubmitHadronsConfig,
              outfile_config_list: OutfileList) -> t.List[str]:
    logging.warning(
        "Check completion succeeds automatically. No implementation of bad_files function in `hadrons_a2a_vectors.py`.")
    return []


def get_task_factory():
    return A2AVecTask.from_dict
