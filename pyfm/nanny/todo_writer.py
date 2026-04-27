import typing as t

from pyfm.nanny.todo import lock_file_name, wait_set_todo_lock, remove_todo_lock


def parse_cfgs(cfg: t.Sequence[int] | None, cfg_range: t.Sequence[int] | None) -> list[str]:
    """Return sorted list of cfgno strings from explicit cfg list or a range."""
    if cfg is not None:
        return [str(c) for c in cfg]
    if cfg_range is not None:
        if len(cfg_range) != 3:
            raise ValueError(
                f"--cfg-range requires exactly 3 integers (start, stop, step), got {len(cfg_range)}"
            )
        start, stop, step = cfg_range
        cfgs = list(range(start, stop + 1, step))
        if not cfgs:
            raise ValueError(
                f"--cfg-range {start} {stop} {step} produces an empty range"
            )
        return [str(c) for c in cfgs]
    raise ValueError("Either --cfg or --cfg-range must be provided")


def validate_steps(steps: t.Sequence[str], job_setup: dict[str, t.Any]) -> None:
    """Raise ValueError if any step name is not a key in job_setup."""
    valid = set(job_setup.keys())
    bad = [s for s in steps if s not in valid]
    if bad:
        raise ValueError(f"Invalid steps: {bad}. Valid: {sorted(valid)}")


def add_entries(
    todo_file: str,
    series: str,
    cfgnos: t.Sequence[str],
    steps: t.Sequence[str],
) -> None:
    """Append one todo entry per cfgno, acquiring the file lock around the write."""
    lock = lock_file_name(todo_file)
    wait_set_todo_lock(lock)
    try:
        with open(todo_file, "a") as f:
            for cfgno in cfgnos:
                parts = [f"{series}.{cfgno}"]
                for step in steps:
                    parts += [step, "0"]
                f.write(" ".join(parts) + "\n")
    finally:
        remove_todo_lock(lock)
