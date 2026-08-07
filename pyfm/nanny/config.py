import typing as t
from enum import auto

from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.core.builder import build_config
from pyfm.domain import SimpleConfig, SerializableEnum, build_hooks
from pyfm.tasks import get_task_handler, list_registered_types


class Scheduler(SerializableEnum):
    LSF = auto()
    PBS = auto()
    SLURM = auto()
    INTERACTIVE = auto()
    COBALT = auto()


@dataclass(frozen=True)
class NannyConfig(SimpleConfig):
    home: str
    todo_file: str
    max_queue: int
    wait: int
    check_interval: int
    job_name_pfx: str
    scheduler: Scheduler


@dataclass(frozen=True)
class JobConfig(SimpleConfig):
    run: str
    job_type: str
    step: str
    io: str
    wall_time: str
    ppn: int
    nodes: int
    lattice: t.List[int]
    geom: t.List[int]
    params: t.Dict[str, t.Any]
    max_cases: int = 1
    tasks: t.Dict[str, t.Any] | None = None
    node_minimum: int | None = None
    task_type: str | None = None
    barrier: bool = True


@dataclass(frozen=True)
class JobBundle:
    """A planned submission: the resolved JobConfig for the active step plus the
    bundled cfgnos to submit in a single job. Returned by ``plan_submission``.
    """

    job_config: JobConfig
    cfgno_steps: t.List[t.List[t.Any]]

    @property
    def ncases(self) -> int:
        return len(self.cfgno_steps)


def get_job_config(job_step: str, yaml_params: t.Dict[str, t.Any]) -> JobConfig:
    job_defaults = yaml_params.get("shared_params", {})
    job_defaults |= {"step": job_step, "params": {}}
    if "job_setup" not in yaml_params:
        raise ValueError("No `job_setup` parameters provided.")
    if job_step not in yaml_params["job_setup"]:
        raise ValueError(f"No `job_setup` parameters provided for `{job_step}`.")

    job_params = job_defaults | yaml_params.get("job_setup").get(job_step)

    job_type, task_type = job_params.get("job_type", None), job_params.get(
        "task_type", None
    )

    if get_task_handler(job_type, task_type) is None:
        raise ValueError(
            f"No task handler found for job_type={job_type!r}, task_type={task_type!r}. "
            f"Registered types: {list_registered_types()}"
        )

    submit = yaml_params.get("submit", {})
    if "resources" not in submit and "layout" not in submit:
        raise ValueError(
            "No `resources` (or legacy `layout`) stanza found under `submit:`."
        )
    if "resources" in submit:
        job_params["resources"] = submit["resources"]
    if "layout" in submit:
        job_params["layout"] = submit["layout"]
    return build_config(JobConfig, job_params)


def build_job_configs(
    yaml_params: t.Dict[str, t.Any],
) -> t.Dict[str, JobConfig]:
    """Build a :class:`JobConfig` for every step defined under ``job_setup``.

    Configs are materialized once so the submission pathway can drive bundling,
    completion checks, and submission off config objects rather than raw primitives.
    Fails fast: a misconfigured step surfaces immediately rather than when it comes up.
    """
    if "job_setup" not in yaml_params:
        raise ValueError("No `job_setup` parameters provided.")
    return {
        step: get_job_config(step, yaml_params) for step in yaml_params["job_setup"]
    }


def normalize_resources(params: t.Dict) -> t.Dict:
    """Canonicalize JobConfig input: rename legacy ``layout`` to ``resources`` and
    flatten the global + per-step resource overrides.

    Accepts the canonical ``resources`` block or the legacy ``layout`` block
    (renamed here for backward compatibility). Scalar top-level keys (e.g.
    ``ppn``, ``max_cases``) are global defaults; the per-step child dict (keyed
    by ``params["step"]``) overrides them. ``max_cases`` defaults to 1 via the
    JobConfig dataclass field when neither layer supplies it. When both
    ``resources`` and ``layout`` are present, ``resources`` wins and a warning
    names it as the canonical key.
    """
    raw = params.pop("resources", None)
    legacy = params.pop("layout", None)
    if raw is not None and legacy is not None:
        utils.get_logger().warning(
            "Both `resources` and legacy `layout` present under `submit:`; "
            "using `resources` and ignoring `layout`."
        )
    if raw is None:
        raw = legacy if legacy is not None else {}

    step = params.get("step")
    global_res = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    per_step = raw.get(step, {}) if (step and isinstance(raw.get(step), dict)) else {}

    return dict(params) | global_res | per_step


def validate_job(config: JobConfig) -> None:
    """Validate JobConfig after construction."""
    if config.max_cases < 1:
        raise ValueError(
            f"max_cases must be >= 1 (step {config.step!r}), got {config.max_cases}"
        )


build_hooks.register(JobConfig, normalize=normalize_resources, validate=validate_job)


def warn_moved_max_cases(params: t.Dict) -> t.Dict:
    """Normalize hook for NannyConfig: warn when the legacy ``nanny.max_cases``
    key is present.

    Must be a ``normalize`` hook (runs on raw params before construction), not
    ``validate`` — pydantic drops unknown keys at construction, so a removed
    ``max_cases`` field is invisible to ``validate``. The legacy value is ignored;
    the user must set ``submit:->resources:->max_cases`` or a per-step override.
    """
    if "max_cases" in params:
        utils.get_logger().warning(
            "`nanny:->max_cases` has moved to " "`submit:->resources:->max_cases`"
        )
    return params


build_hooks.register(NannyConfig, normalize=warn_moved_max_cases)


def get_nanny_config(yaml_params: t.Dict[str, t.Any]) -> NannyConfig:
    nanny_params = yaml_params.get("shared_params", {})
    nanny_params |= yaml_params["nanny"]
    nanny_params |= yaml_params["submit"]
    nanny_params |= yaml_params.get("files", {})
    return build_config(NannyConfig, nanny_params)
