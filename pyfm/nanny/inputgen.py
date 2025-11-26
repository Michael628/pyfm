import typing as t
import yaml

from pyfm.nanny.setup import create_task
from pyfm import utils
from pyfm.domain import Outfile
from pyfm.tasks.hadrons import hadmods


@t.runtime_checkable
class InputGeneratorProtocol(t.Protocol):
    def build_input_params(self) -> t.Any: ...
    def format_string(self, to_format: str) -> str: ...

    @property
    def key(self) -> str: ...


def write_input_file(job_step: str, yaml_data: t.Dict, series: str, cfg: str) -> str:
    task: InputGeneratorProtocol = create_task(job_step, yaml_data, series, cfg)

    infile_template = yaml_data["job_setup"][job_step]["io"]
    infile_template += "-{series}.{cfg}"
    infile_stem = task.format_string(infile_template)

    task_key = task.key
    if "smear" in task_key:
        infile = utils.io.write_plain_text(
            infile_stem, task.build_input_params(), ext="txt"
        )
    elif "hadrons" in task_key:
        hadrons_input = task.build_input_params()

        schedule_file = utils.io.write_schedule(infile_stem, hadrons_input.schedule)
        xml_dict = hadmods.xml_wrapper(
            runid=task.config.runid, sched=schedule_file, cfg=cfg
        )
        modules = list(hadrons_input.modules.values())
        xml_dict["grid"]["modules"] = {"module": modules}
        infile = utils.io.write_xml(infile_stem, xml_dict)
    elif "contract" in task_key:
        yaml.add_representer(Outfile, lambda d, x: d.represent_dict(x.__dict__))
        input_params = yaml.dump(task.build_input_params())
        infile = utils.io.write_plain_text(infile_stem, input_params, ext="yaml")
    # TODO: use for raw hadrons task
    # if "xml_file" in input_params:
    #     with open(input_params["xml_file"], "r") as f:
    #         input_string = f.read()

    else:
        raise NotImplementedError(f"Write input file not implemented for {job_step}")

    return infile
